from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import os
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
import argparse
import numpy as np
from transformers import get_linear_schedule_with_warmup
from model.utils import set_seed
# from misc_files.utils import *

def load_data_for_training(
        tokenizer,
        loader_path,
        dataset_dir,
        max_input_length=256,
    ):

    def preprocess_function(examples):
        inputs = [doc for doc in examples["text"]]
        model_inputs = tokenizer(
            inputs, max_length=max_input_length, truncation=True
        )
        # input_ids = model_inputs["input_ids"]
        return model_inputs

    # preprocess dataset
    datasets = load_dataset(
        path=loader_path,
        data_dir=dataset_dir,
        # download_mode="force_redownload", # to force redownload
    )
    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    return tokenized_datasets

class Collate_fn:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self,batch):
        
        input_ids = [torch.tensor(x["input_ids"]) for x in batch]
        input_ids = pad_sequence(input_ids, batch_first=True,padding_value = self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        label = torch.stack([torch.tensor(x["label"]) for x in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }


def get_loss(batch,model,loss_fn,args,training=False,return_logits = False,nli_model=False):
    inp = batch["input_ids"]
    mask = batch["attention_mask"]
    labels = batch["label"]
    # print (labels)
    if not nli_model:
        if not training:
            with torch.no_grad():
                pred_logits = model(inp,mask)
        else:
            pred_logits = model(inp,mask)
    else:
        if not training:
            with torch.no_grad():
                out = model(input_ids = inp,attention_mask = mask)
                pred_logits = out.logits
        else:
            out = model(input_ids = inp,attention_mask = mask)
            pred_logits = out.logits
    loss = loss_fn(pred_logits,labels)
    if return_logits:
        return loss,pred_logits
    else:
        return loss

def build_optimizer(model, predictor = None,optimizer_name="adam", learning_rate=1e-5):
    param_optimizer = list(model.named_parameters())
    if predictor is not None:
        param_optimizer += list(predictor.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if optimizer_name == "adam":
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
    else:
        assert False, "optimizer_name = '%s' is not `adam` or `lamb`" % (optimizer_name)
    return optimizer    

"""
Minimal training script to train a NLI model to predict NLI labels corresponding to rationales

grad_norm = 1.0
learning_rate = 2e-5
lr scheduler

"""

def train_nli(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.encoder_model_id,num_labels = 3).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_id)
    
    optimizer = build_optimizer(model,optimizer_name='adam',learning_rate=args.learning_rate)
    if args.answer_only:
        data_path = f"data/{args.dataset_name}_answer"
    # elif args.pct_train_rationales != 0.1:
    else:
        data_path = f"data/{args.dataset_name}/nli_{int(args.pct_train_rationales*100)}"
    dataset = load_data_for_training(tokenizer,"preprocess/nli_loader.py",data_path,max_input_length=args.max_length)
    loss_fn = nn.CrossEntropyLoss()
    collate_fn = Collate_fn(tokenizer)
    train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size,shuffle = True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset['validation'], batch_size=args.batch_size,shuffle = False, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset['test'], batch_size=args.batch_size,shuffle = False, collate_fn=collate_fn)
    
    total_steps = len(train_dataloader) * 10  # 10 here is the number of epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    log_dir = f'results/nli_predictor'
    # if args.answer_only:
    #     log_dir += '_answer'
    model_dir = 'nli_weights'
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    patience = 0

    best_val_loss = 1e13
    print ('Starting training')
    for epoch in tqdm(range(10),total = 10,desc = f'Training nli predictor for {args.dataset_name}'):
        model.train()
        epoch_train_loss = []
        for b_no,batch in tqdm(enumerate(train_dataloader),total = len(train_dataloader),desc = f'Epoch {epoch}'):
            optimizer.zero_grad()
            batch = {k:v.to(args.device) for k,v in batch.items()}
            loss = get_loss(batch,model,loss_fn,args,training=True,nli_model=True)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            epoch_train_loss.append(loss.item())

        ## Eval
        model.eval()
        total_val_loss = []
        for v_batch in tqdm(val_dataloader,desc = f'Validating nli predictor for {args.dataset_name}',total = len(val_dataloader)):
            v_batch = {k:v.to(args.device) for k,v in v_batch.items()}
            v_loss = get_loss(v_batch,model,loss_fn,args,training=False,nli_model=True)
            total_val_loss.append(v_loss.item())


        mean_loss = np.mean(total_val_loss)
        
        with open(os.path.join(log_dir,f'{args.dataset_name}_{int(args.pct_train_rationales*100)}.txt'),'a') as f:
            f.write('epoch: {} train loss:{:.3f}\n'.format(epoch,np.mean(epoch_train_loss)))
            f.write('epoch: {} val loss:{:.3f}\n'.format(epoch,mean_loss))
        print ('epoch: {} train loss:{:.3f} val_loss: {:.3f}'.format(epoch,np.mean(epoch_train_loss),mean_loss))
        if mean_loss < best_val_loss:
            patience = 0
            best_val_loss = mean_loss
            model_state_dict = model.state_dict()
            if args.encoder_model_id.split('/')[0] != 'cross-encoder':
                add_name = 'untrained'
            else:
                add_name = 'trained'
            
            
            if args.answer_only:
                torch.save(model_state_dict,os.path.join(model_dir,f'{args.dataset_name}_nli_model_answer_{add_name}.pt'))
            else:
                torch.save(model_state_dict,os.path.join(model_dir,f'{args.dataset_name}_nli_model_{int(args.pct_train_rationales*100)}_{add_name}.pt'))
            print('saved best model')
        else:
            patience += 1
        
        if patience >= 2:
            print ('early stopping')
            break
    
    test_acc,test_f1 = [],[]
    for test_batch in tqdm(test_dataloader,desc = f'Testing nli predictor for {args.dataset_name}',total = len(test_dataloader)):
        test_batch = {k:v.to(args.device) for k,v in test_batch.items()}
        _,test_logits = get_loss(test_batch,model,loss_fn,args,training=False,return_logits=True,nli_model=True)
        true_labels = test_batch['label']
        pred_labels = torch.argmax(torch.softmax(test_logits,dim=-1),dim = -1)
        test_acc.append(accuracy_score(true_labels.cpu().numpy(),pred_labels.cpu().numpy()))
        test_f1.append(f1_score(true_labels.cpu().numpy(),pred_labels.cpu().numpy(),average='macro'))

    with open(os.path.join(log_dir,f'{args.dataset_name}_{int(args.pct_train_rationales*100)}.txt'),'a') as f:
        f.write('test acc:{:.2f}\n'.format(np.mean(test_acc)))
        f.write('test f1:{:.2f}\n'.format(np.mean(test_f1)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_model_id",type=str, required=False, default = 'cross-encoder/nli-deberta-v3-large')
    parser.add_argument("--device",type=str, required=False, default = 'cuda:0')
    parser.add_argument("--dataset_name",type=str, required=False, default = 'multirc')
    parser.add_argument("--max_length",type=int, required=False, default = 512)
    parser.add_argument("--learning_rate",type=float, required=False, default = 2e-5)
    parser.add_argument("--seed",type=int, required=False, default = 42)
    parser.add_argument('--answer_only', type=bool, default=False, help='for Q&A dataset, take only answer')
    parser.add_argument('--pct_train_rationales', type=float, default=0.1, help='Percentage of train examples to provide gold rationales for. None means all available train examples are used.')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    set_seed(args.seed)

    train_nli(args)

if __name__ == "__main__":
    main()