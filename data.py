import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
from torch.utils.data import Sampler,RandomSampler,DistributedSampler
# from torchsampler import ImbalancedDatasetSampler
from time import time
from multiprocessing import Process, Manager
import pickle
from preprocess.data import *
import pandas as pd
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import random


class TokenClassificationDataset(Dataset):
    def __init__(self,tokenizer,data_dir,task,split,max_length,nli_tokenizer,debug=False,answer_only=False,attack=False,pct_data=1.0):
        self.split = split
        self.tokenizer = tokenizer
        self.nli_tokenizer = nli_tokenizer
        self.data_dir = data_dir
        self.task = task
        self.max_length = max_length
        self.pad_id = self.tokenizer.pad_token_id
        self.debug = debug
        self.answer_only = answer_only
        self.attack = attack
        self.pct_data = pct_data
        self._setup()
    
    def __len__(self):
        return len(self.data['input_ids'])

    def _setup(self):

        self.data = {}
        task_dir = os.path.join(self.data_dir,self.task)
        if self.answer_only:
            main_dir = os.path.join(task_dir,f'{self.task}_{self.split}_answer.pkl')
        else:
            main_dir = os.path.join(task_dir,f'{self.task}_{self.split}.pkl')
            
        if self.split == 'test' and self.attack: # during testing evaluate robustness on attack instances, override main_dir
            main_dir = os.path.join(task_dir,f'{self.task}_{self.split}_attack.pkl') 
        
        with open(main_dir,'rb') as f:
            loaded_data = pickle.load(f)
        
        if self.pct_data < 1.0:
            sample_size = int(self.pct_data*len(loaded_data['input_ids']))
            # sample random indices
            indices = np.random.choice(np.arange(len(loaded_data['input_ids'])),sample_size,replace=False)
            for k,v in loaded_data.items():
                loaded_data[k] = [v[i] for i in indices]
        self.data['label'] = [d for d in loaded_data['label']] # batch size
        self.data['z_gold'] = [d for d in loaded_data['rationale']] ## needs to pad (during collate)
        sen_spans = [d for d in loaded_data['sentence_span']]
        self.data['sen_start'] = [[ss[0] for ss in d] for d in sen_spans]
        self.data['sen_end'] = [[ss[1] for ss in d] for d in sen_spans]
        self.data['query'] = [d for d in loaded_data['query']]
        self.data['input_ids'] = [d for d in loaded_data['input_ids']]
        if self.attack:
            self.data['noisy_z'] = [d for d in loaded_data['noisy_rationale']]
        
        if self.debug: # just for debugging
            for k,v in self.data.items():
                self.data[k] = v[:100]

  
    def __getitem__(self, index):
        input_ids = self.data['input_ids'][index].long() ## need pad
        z_gold = self.data['z_gold'][index].long() ## need pad
        sen_start = torch.tensor(self.data['sen_start'][index]).long() ## need pad
        sen_end = torch.tensor(self.data['sen_end'][index]).long() ## need pad
        label = torch.tensor(self.data['label'][index]).long()
        query = self.data['query'][index]
        if self.attack:
            noisy_z = self.data['noisy_z'][index].long()
            return {
            'input_ids': input_ids,
            'z_gold': z_gold,
            'sen_start': sen_start,
            'sen_end': sen_end,
            'label': label,
            'query': query,
            'noisy_z': noisy_z
            }
        return {
        'input_ids': input_ids,
        'z_gold': z_gold,
        'sen_start': sen_start,
        'sen_end': sen_end,
        'label': label,
        'query': query
        }

    def collate_fn(self, batch):
        task = self.task
        out_dict = {}

        inps = [b['input_ids'] for b in batch]
        z_gold = [b['z_gold'] for b in batch]
        sen_start = [b['sen_start'] for b in batch]
        sen_end = [b['sen_end'] for b in batch]
        query = [b['query'] for b in batch]
        if self.attack:
            noisy_z = [b['noisy_z'] for b in batch]
            out_dict['noisy_z'] = pad_sequence(noisy_z,batch_first=True,padding_value=0)
        # max_num_sen = max([ss.shape[0] for ss in sen_start])

        out_dict['input_ids'] = pad_sequence(inps,batch_first=True,padding_value=self.pad_id)
        out_dict['attention_mask'] = (out_dict['input_ids'] != self.pad_id).long()
        out_dict['z_gold'] = pad_sequence(z_gold,batch_first=True,padding_value=0)
        out_dict['sen_start'] = pad_sequence(sen_start,batch_first=True,padding_value=0)
        sen_end = pad_sequence(sen_end,batch_first=True,padding_value=-1)
        out_dict['sen_mask'] = (sen_end != -1).long() # batch size, max sentences
        out_dict['sen_end'] = torch.clamp(sen_end,min=1) # set to 1 to do retrieval, mask will take care of it
        out_dict['label'] = torch.stack([b['label'] for b in batch])

        ## need to process nli inputs
        nli_inps,nli_mask = self.get_nli_inputs(sen_start,sen_end,inps,query)
        out_dict['nli_inputs'] = nli_inps
        out_dict['nli_mask'] = nli_mask
        self.check_len(out_dict['sen_start'],out_dict['sen_end'],nli_inps)
        
        return out_dict
    
    def get_nli_inputs(self,sen_start,sen_end,inputs,query):
        max_sentences = max([ss.shape[0] for ss in sen_start])
        batch_size = len(sen_start)
        
        all_nli_inps = []
        for b in range(batch_size):
            curr_nli_inputs = []
            curr_query = query[b]
            for ss,se in zip(sen_start[b],sen_end[b]):
                curr_nli_inputs.append(inputs[b].gather(dim=0,index=torch.arange(ss,se))) # includes cls token
            curr_nli_text = self.tokenizer.batch_decode(curr_nli_inputs,skip_special_tokens=True)
            curr_inp_text = [curr_query + ' ' + t for t in curr_nli_text]
            curr_nli_enc = self.nli_tokenizer.batch_encode_plus(curr_inp_text,add_special_tokens=True,padding='longest',return_tensors = 'pt')['input_ids']
            all_nli_inps.append(curr_nli_enc) # (num_sents,seq_len)
        max_length = max([x.size(1) for x in all_nli_inps])
        padded_sen_len = [torch.nn.functional.pad(tensor, (0, max_length - tensor.size(1)),value = self.nli_tokenizer.pad_token_id) for tensor in all_nli_inps] ## pad to max sentence length
        padded_nli_inps = [torch.nn.functional.pad(tensor, (0,0,0, max_sentences - tensor.size(0)),value = self.nli_tokenizer.pad_token_id) for tensor in padded_sen_len] ## pad to max sentences
        padded_nli_inps = torch.stack(padded_nli_inps,dim=0)
        nli_mask = (padded_nli_inps != self.nli_tokenizer.pad_token_id).long()
        assert padded_nli_inps.shape[1] == max_sentences, 'max sentences dont match'
        return padded_nli_inps,nli_mask
    
    def check_len(self,ss,se,nli_inps):
        assert ss.shape[0] == se.shape[0] == nli_inps.shape[0], 'batches of sentences dont match'
        
                    
class TaskDataLoader:
    """Wrapper around dataloader to keep the task names."""
    def __init__(self, task, dataset, batch_size=8,
                 collate_fn=None, drop_last=False,
                 num_workers=0, sampler=None,n_gpu=1):
        self.dataset = dataset
        self.task = task
        self.batch_size = batch_size 
        if sampler is None and n_gpu > 1:
            sampler = DistributedSampler(dataset)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      collate_fn=collate_fn,
                                      drop_last=drop_last,
                                      num_workers=num_workers)
    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task"] = self.task
            yield batch


# Note not to use itertools.cycle since it is
# doing some caching under the hood, resulting
# in issues in the dataloading pipeline.
# see https://docs.python.org/3.7/library/itertools.html?highlight=cycle#itertools.cycle
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class MultiTaskDataLoader:
    """Given a dictionary of task: dataset, returns a multi-task dataloader
    which uses temperature sampling to sample different datasets."""

    def __init__(self,  max_steps, tasks_to_datasets, batch_size=8, collate_fn=None,
                 drop_last=False, num_workers=0, temperature=100.0,n_gpu=1):
        # Computes a mapping from task to dataloaders.
        self.task_to_dataloaders = {}
        for task, dataset in tasks_to_datasets.items():
            dataloader = TaskDataLoader(task, dataset, batch_size,
                collate_fn=collate_fn, drop_last=drop_last, num_workers=num_workers,n_gpu=n_gpu)
            self.task_to_dataloaders.update({task: dataloader})
        self.tasknames = list(self.task_to_dataloaders.keys())

        # Computes the temperature sampling weights.
        self.sampling_weights = self.temperature_sampling(self.dataloader_sizes.values(), temperature)
        self.dataiters = {k: cycle(v) for k, v in self.task_to_dataloaders.items()}
        self.max_steps = max_steps

    def temperature_sampling(self, dataset_sizes, temp):
        total_size = sum(dataset_sizes)
        weights = np.array([(size / total_size) ** (1.0 / temp) for size in dataset_sizes])
        return weights/np.sum(weights)

    @property
    def dataloader_sizes(self):
        if not hasattr(self, '_dataloader_sizes'):
            self._dataloader_sizes = {k: len(v) for k, v in self.task_to_dataloaders.items()}
        return self._dataloader_sizes

    def __len__(self):
        return sum(v for k, v in self.dataloader_sizes.items())

    def num_examples(self):
        return sum(len(dataloader.dataset) for dataloader in self.task_to_dataloaders.values())

    def __iter__(self):
        for i in range(self.max_steps):
            taskname = np.random.choice(self.tasknames, p=self.sampling_weights)
            dataiter = self.dataiters[taskname]
            outputs = next(dataiter)
            yield outputs
    




      
    
        
        

        
        