import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from preprocess.eraser_utils import *
import argparse
from itertools import chain
from data import dataset_info
import pickle
import numpy as np
import torch
import random
from model.utils import preprocess_query,tfidf_selection
from sklearn.feature_extraction.text import TfidfVectorizer

def makefile(file_path):
    if not os.path.exists(file_path):
        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass

def check_esnli(dataline):
    all_docids = []
    for ee in dataline['evidences']:
        for e in ee:
            all_docids.append(e['docid'])
    all_docids = list(set(all_docids))
    doc_labels = [d.rsplit('_',1)[-1] for d in all_docids]
    if 'premise' in doc_labels and 'hypothesis' in doc_labels:
        return True
    else:
        return False

def save_dataset(data_path, dataset_dict, split):
    for key in tqdm(dataset_dict.keys(), desc=f'Saving {split} dataset'):
        filename = f'{key}.pkl' 
        with open(os.path.join(data_path, filename), 'wb') as f:
            pickle.dump(dataset_dict[key], f)

def word_to_token_mask(tokenizer,text,mask,reverse=False):
    num_text = len(text.split())
    tokenized = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokenized_text = tokenized.input_ids
    offsets = tokenized.offset_mapping
    aligned_mask = [mask[0]] # add 1st value in 
    
    end = offsets[0][1]
    mask_pointer = 0
    for offset in offsets[1:]:
        if offset[0] > end: # new word
            mask_pointer += 1
        else:
            if reverse: # reverse used to convert token to word level.
                end = offset[1]
                continue
        end = offset[1]
        aligned_mask.append(mask[mask_pointer])
    if reverse:
        assert len(aligned_mask) == num_text, f'{len(aligned_mask)},{num_text}'
    else:
        assert len(aligned_mask) == len(tokenized_text), f'{len(aligned_mask)},{len(tokenized_text)}'
    return aligned_mask

def contains_unicode_in_range(text, start, end):
    for char in text:
        if start <= ord(char) <= end:
            return True
    return False

def check_duplicates(list_of_dicts, key):
    # Extract values of the specified key from all dictionaries
    values = [d[key] for d in list_of_dicts if key in d]

    # Check for duplicates
    if len(values) != len(set(values)):
        print (len(values),len(set(values)))
        return False
    else:
        return True
    



def get_query_length(query,task,tokenizer):
    if task == 'multirc':
        split_query = query.split('||')
        query = ' '.join(split_query)
    elif task == 'evidence_inference':
        split_query = query.split('|')
        query = ' '.join(split_query)
    query_len = len(tokenizer.tokenize(query))
    return query_len


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.arch)
    tfidfvec = TfidfVectorizer(use_idf=True)
    # model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-large')
    random.seed(args.seed)
    sep_token = tokenizer.sep_token
    all_fil = ['train','val','test']
    for task in args.dataset: # which task, boolq,...
        documents_path = os.path.join(args.data_dir, task,'documents.pkl')
        classes = dataset_info[task]['classes']
        print (f'Processing {task}')
        task_dir = os.path.join(args.data_dir, task)
        abs_path= os.path.abspath(task_dir)
        if os.path.exists(documents_path):
            print (f'{documents_path} exists, loading...')
            documents = torch.load(documents_path)
        else:
            documents = load_documents(abs_path,split_sentence=False)
            torch.save(documents, documents_path)

        for file in all_fil: # train, val
            if args.answer_only:
                new_data_directory = f"{task_dir}_answer"
            else:
                new_data_directory = os.path.join(task_dir,f'nli_{int(args.pct_train_rationales*100)}')
            if not os.path.exists(new_data_directory):
                os.makedirs(new_data_directory)
                
            new_datadir = os.path.join(new_data_directory,f'nli_{file}.jsonl')
            
            dataset_dict = []
            
            curr_file = os.path.join(task_dir, file + '.jsonl')
            with open(curr_file, 'r',encoding='utf-8',errors = 'ignore') as f:
                old_dataset = [json.loads(l) for l in f]
            
            random.shuffle(old_dataset)
            stop_len = int(args.pct_train_rationales*len(old_dataset)) # use 10%
            
            original_datalen = len(old_dataset)
            
            data_count = 0
            processed = 0
            for old_d in tqdm(old_dataset,desc = f' getting {file}', total= original_datalen ): # each line in train.jsonl,etc..
                task_label  = old_d['classification']
                label = classes.index(task_label)
                query = old_d['query']
                query = preprocess_query(query,task,answer_only = args.answer_only)
                query_len = len(tokenizer.tokenize(query))
                ## CHECKS
                if len(task_label) < 1:
                    continue
                go_ahead = True
                if task == 'esnli' and task_label != 'neutral':
                    go_ahead = check_esnli(old_d) # take away certain datapoints with either premise or hypothesis only
                if go_ahead:
                    if len(old_d['evidences']) < 1:
                        continue
                    docids = list(set(ev['docid'] for ev in chain.from_iterable(old_d['evidences'])))
                    evs = chain.from_iterable(old_d['evidences'])
                    evidence_len = [len(tokenizer.tokenize(ev['text'])) for ev in chain.from_iterable(old_d['evidences'])]
                    for el in evidence_len:
                        if el > (args.max_length - query_len - 2)*0.75: # too long evidence
                            continue
                        
                    for docid in docids[:1]:
                        doc = documents[docid]
                        if task == 'evidence_inference' or task == 'boolq': # truncate using tf-idf
                            tokenized_sen = [sen.split(' ') for sen in doc]
                            rationale_span = set([(ev['start_sentence'],ev['end_sentence']+1) for ev in evs])
                            doc,rationale_sen = tfidf_selection(tfidfvec,tokenized_sen,query,rationale_span,max_sen = 20)
                            if doc == None: # more than 20 sentences
                                continue
                            rationale_sen = set(sum([np.arange(rs[0],rs[1]).tolist() for rs in rationale_sen],[])) # add one 
                        else:
                            rationale_sen = set([np.arange(ev['start_sentence'],ev['end_sentence']).item() for ev in evs])
                            
                        for s_id,sen in enumerate(doc):
                            if s_id in rationale_sen:
                                if label == 0:
                                    nli_label = 0
                                elif label == 1:
                                    nli_label = 1
                                else: # if there are 3 labels in the case of evidence inference
                                    nli_label = 2 # no significant difference can be set to neutral
                            else: # assumping that non-rationales has no contradictions (should have small amount of data)
                                nli_label = 2
                            
                            curr_text = query + ' ' + sen # train each sentence to match against the query
                            curr_d = {}
                            curr_d['text'] = curr_text
                            curr_d['label'] = nli_label
                            # curr_d['query_span'] = span
                            curr_d['id'] = data_count
                            data_count += 1
                            dataset_dict.append(curr_d)
                    processed += 1
                    if processed >= stop_len and (file == 'train' or file == 'val'):
                        break


            assert check_duplicates(dataset_dict,'id'), 'Duplicates found in dataset'
            with open(new_datadir, 'w') as f:
                for d in dataset_dict:
                    json.dump(d, f)
                    f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--data_dir', type=str, default='data', help='Root directory for datasets')
    parser.add_argument('--dataset', type=str,
                        choices=['cose', 'esnli', 'movies', 'multirc', 'sst', 'amazon', 'yelp', 'stf', 'olid', 'irony','fever','evidence_inference','boolq'],nargs = '+')
    parser.add_argument('--arch', type=str, default='cross-encoder/nli-deberta-v3-large')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of examples to sample. None means all available examples are used.')
    parser.add_argument('--pct_train_rationales', type=float, default=0.1, help='Percentage of train examples to provide gold rationales for. None means all available train examples are used.')
    parser.add_argument('--max_length', type=int, default=512, )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--answer_only', type=bool, default=False, help='for Q&A dataset, take only answer')

    args = parser.parse_args()
    main(args)