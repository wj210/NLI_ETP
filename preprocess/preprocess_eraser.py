import json
import os
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from preprocess.eraser_utils import *
import argparse
from itertools import chain
from data import dataset_info
import pickle
import numpy as np
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from model.utils import preprocess_query,tfidf_selection


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

def contains_unicode_in_range(sentence, start, end):
    for text in sentence:
        for char in text:
            if start <= ord(char) <= end:
                return True
    return False

def get_max_sentences(tokenized_text,max_length,take_pos= 'start',start_pos =0):
    curr_len = 0
    if take_pos == 'start':
        for i,sent in enumerate(tokenized_text):
            curr_len += len(sent)
            if curr_len > max_length:
                return (0,i)
    elif take_pos == 'end':
        for i,sen in enumerate(reversed(tokenized_text)):
            curr_len += len(sen)
            if curr_len > max_length:
                return (len(tokenized_text)-i+1,len(tokenized_text))
    elif take_pos == 'middle':
        for i,sen in enumerate(tokenized_text[start_pos:]):
            curr_len += len(sen)
            if curr_len > max_length:
                return (start_pos,start_pos+i)
        return (start_pos,start_pos+i)
                
    else:
        raise ValueError(f'Invalid take_pos {take_pos}') 
    

def check_condition(rationale,list_pos,evidence_is_sequence=False):
    if evidence_is_sequence:
        rationale_chk = ''.join(map(str,rationale))
        against = ''.join(map(str,list(range(list_pos[0],list_pos[1]))))
        check_condition = rationale_chk in against
    else:
        check_condition = rationale[0] in list(range(list_pos[0],list_pos[1]))
    return check_condition


def truncate_inputs(tokenized_text,rationale,tokenizer,max_length,evidence_is_sequence=False):
    start_pos,end_pos = get_max_sentences(tokenized_text,max_length,take_pos='start')
    if check_condition(rationale,(start_pos,end_pos),evidence_is_sequence): # in start
        truncated_inp = tokenized_text[start_pos:end_pos]
        out_pos = [start_pos,end_pos]
    else:
        ## try from back
        start_pos,end_pos = get_max_sentences(tokenized_text,max_length,take_pos='end')
        if check_condition(rationale,(start_pos,end_pos),evidence_is_sequence):
            truncated_inp = tokenized_text[start_pos:end_pos]
            out_pos = [start_pos,end_pos]

        else: ## take from middle
            if not evidence_is_sequence:
                starting_pos = rationale[0] 
                start_pos = max(0,starting_pos-5) # set 5 as middle ground to start
            else:
                start_pos = rationale[0][0]

            start_pos,end_pos = get_max_sentences(tokenized_text,max_length,take_pos='middle',start_pos=start_pos)
            truncated_inp = tokenized_text[start_pos:end_pos]
            out_pos = [start_pos,end_pos]
    
    if evidence_is_sequence:     
        rationale = sum(rationale,[]) # unroll it , might take half rationales, eitherway sentences are truncated.
            
    out_rationale = []
    actual_rationale = []
    for i in range(out_pos[0],out_pos[1]):
        if i in rationale:
            # out_rationale.append(rationale.index(i))
            out_rationale.append(i- out_pos[0])
            actual_rationale.append(i)
    out_rationale = sorted(out_rationale)
    if len(out_rationale) < 1: # unclear why
        return None,None,None 
    assert out_rationale[-1] < len(truncated_inp), f'{out_rationale[-1]},{len(truncated_inp)},{rationale},{out_pos}'
    assert sum([len(r) for r in truncated_inp]) <= max_length, f'{sum(len(r) for r in truncated_inp)},{max_length}'
    
    return truncated_inp,out_rationale,actual_rationale
    

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
    if args.attack:
        all_fil = ['test'] # only attack the test set.
    else:
        all_fil = ['train','val','test']
    for task in args.dataset: # which task, boolq,...
        if args.attack:
            documents_path = os.path.join(args.data_dir, task,'attack_documents.pkl')
        else:
            documents_path = os.path.join(args.data_dir, task,'documents.pkl')
        num_special_tokens = dataset_info[task]['num_special_tokens']
        max_length = dataset_info[task]['max_length'][args.arch]
        classes = dataset_info[task]['classes']
        print (f'Processing {task}')
        task_dir = os.path.join(args.data_dir, task)
        abs_path= os.path.abspath(task_dir)
        max_len = 0
        if os.path.exists(documents_path):
            print (f'{documents_path} exists, loading...')
            documents = torch.load(documents_path)
        else:
            documents = load_documents(abs_path,split_sentence = False,attack=args.attack)
            torch.save(documents, documents_path)

        max_len = max([len(d) for d in documents.values()])
        avg_len = np.mean([len(d) for d in documents.values()]).item()
        print (f'Maximum number of sentences: {max_len}')
        print (f'Average number of sentences: {avg_len}')
        for file in all_fil: # train, val
            if args.answer_only:
                new_datadir = os.path.join(task_dir,f'{task}_{file}_answer.pkl')
            elif args.attack:
                new_datadir = os.path.join(task_dir,f'{task}_{file}_attack.pkl')
            else:
                new_datadir = os.path.join(task_dir,f'{task}_{file}.pkl')
            dataset_dict = defaultdict(list)
            
            if args.attack:
                curr_file = os.path.join(task_dir, f'attack_{file}.jsonl')
            else:
                curr_file = os.path.join(task_dir, file + '.jsonl')
            with open(curr_file, 'r',encoding='utf-8',errors = 'ignore') as f:
                old_dataset = [json.loads(l) for l in f]
            
            original_datalen = len(old_dataset)
            for old_d in tqdm(old_dataset,desc = f' getting {file}', total= original_datalen ): # each line in train.jsonl,etc..
                task_label  = old_d['classification']
                query = old_d['query']
                augs = old_d.get('augs',None)
                query = preprocess_query(query,task,answer_only = args.answer_only) # preprocess query
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

                    for doc in docids[:1]:
                        label = classes.index(task_label)
                        input_text = documents.get(doc,None) # sentences
                        if input_text is None: # happens during attack, certain document cannot be found due to error in the robustness attack code.
                            continue 
                        input_text = [re.sub(' +',' ',it) for it in input_text]
                        if task in ['boolq', 'evidence_inference']:
                            is_sequence = True
                            tokenized_inp = [sen.split(' ') for sen in input_text]
                            rationale_span = set([(ev['start_sentence'],ev['end_sentence']+1) for ev in evs])
                            input_text,rationale_sen = tfidf_selection(tfidfvec,tokenized_inp,query,rationale_span,max_sen = 20)
                            if input_text == None: # more than 20 sentences
                                continue
                            rationale_sen = [np.arange(rs[0],rs[1]).tolist() for rs in rationale_sen]

                        else:
                            rationale_sen = list(set([np.arange(ev['start_sentence'],ev['end_sentence']).item() for ev in evs]))
                            is_sequence = False
                        
                        rationale_sen = sorted(rationale_sen)
                        all_tokenized_text = [tokenizer(inp,add_special_tokens=False)['input_ids'] for inp in input_text]
                        inp_length = sum([len(inp) for inp in all_tokenized_text])

                        all_sentence_spans = []
                         # encoding error
                        if contains_unicode_in_range(input_text,0x80, 0xFF):
                            continue
                        
                        allowable_length = max_length - num_special_tokens 
                        if task in ['fever','multirc','boolq', 'evidence_inference']:
                            allowable_length -= query_len
                        if inp_length > allowable_length: # if exceeds max length
                            all_tokenized_text,rationale_sen,actual_rationale_sen = truncate_inputs(all_tokenized_text,rationale_sen,tokenizer,allowable_length,evidence_is_sequence=is_sequence)
                            if all_tokenized_text is None:
                                continue
                            # if input_text is None:
                            #     continue


                        tokenized_query = tokenizer(query,add_special_tokens=False)['input_ids']

                        if isinstance(rationale_sen[0],list): # if is a list of list
                            rationale_sen = sum(rationale_sen,[])
                        
                        ## setup sentence rationale
                        out_sentence_rationale = [0]* len(all_tokenized_text)
                        if augs is not None:
                            aug_rationale = [0]* len(all_tokenized_text)
                            for aug in augs:
                                aug_rationale[aug] = 1

                        for r_sen in rationale_sen:
                            out_sentence_rationale[r_sen] = 1
                        
                        ## setup sentence spans
                        starting_len = len(tokenized_query) + 1 # CLS token
                        all_tokens = []
                        for sen_no,tokenized_sen in enumerate(all_tokenized_text):
                            all_tokens.extend(tokenized_sen)
                            all_sentence_spans.append([starting_len,starting_len+len(tokenized_sen)])
                            starting_len += len(tokenized_sen)
                        

                        all_tokens = tokenized_query + all_tokens
                        assert all_sentence_spans[-1][-1]-1 <= len(all_tokens), f'{all_sentence_spans[-1][-1]},{len(all_tokens)}'
                        
                        all_tokens = [tokenizer.cls_token_id] + all_tokens + [tokenizer.sep_token_id]

                        assert len(all_tokens) <= max_length, f'{len(all_tokens)},{max_length}'
                        
                        dataset_dict['input_ids'].append(torch.tensor(all_tokens))
                        dataset_dict['rationale'].append(torch.tensor(out_sentence_rationale))
                        dataset_dict['label'].append(label)
                        dataset_dict['sentence_span'].append(all_sentence_spans)
                        dataset_dict['query'].append(query)
                        if augs is not None:
                            dataset_dict['noisy_rationale'].append(torch.tensor(aug_rationale))

            with open(new_datadir, 'wb') as f:
                print ('old dataset: {}, new_dataset: {}'.format(original_datalen,len(dataset_dict['input_ids'])))
                pickle.dump(dataset_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--data_dir', type=str, default='data', help='Root directory for datasets')
    parser.add_argument('--dataset', type=str,
                        choices=['cose', 'esnli', 'movies', 'multirc', 'sst', 'amazon', 'yelp', 'stf', 'olid', 'irony','fever','evidence_inference','boolq'],nargs = '+')
    parser.add_argument('--arch', type=str, default='roberta-base')
    parser.add_argument('--split', type=str, help='Dataset split', choices=['train', 'val', 'test'])
    parser.add_argument('--num_samples', type=int, default=None, help='Number of examples to sample. None means all available examples are used.')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--answer_only', type=bool, default=False, help='for Q&A dataset, take only answer')
    parser.add_argument('--attack', type=bool, default=False, help='for robustness')
    args = parser.parse_args()
    main(args)