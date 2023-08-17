from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchtext.vocab import GloVe,vocab,Vectors
from collections import Counter

from transformers import RobertaModel
import json

class FSDPModel(pl.LightningModule):
    def __init__(self,enc_model,sen_head,tok_head,task_head):
        super().__init__()
        self.enc_model = enc_model
        self.sen_head = sen_head
        self.tok_head = tok_head
        self.task_head = task_head

    def forward(self, input_ids,attention_mask = None,task = 'encode'):
        if task == 'encode':
            out = self.encoder_model(input_ids,attention_mask=attention_mask)
        elif task == 'sen':
            out = self.sen_model(input_ids)
        elif task == 'token':
            out = self.tok_head(input_ids)
        elif task == 'task':
            out = self.task_head(input_ids)
        return out
    
def MLP_factory(layer_sizes, dropout=False, layernorm=False):
    modules = nn.ModuleList()
    unpacked_sizes = []
    for block in layer_sizes:
        unpacked_sizes.extend([block[0]] * block[1])

    for k in range(len(unpacked_sizes)-1):
        if layernorm:
            modules.append(nn.LayerNorm(unpacked_sizes[k]))
        modules.append(nn.Linear(unpacked_sizes[k], unpacked_sizes[k+1]))
        if k < len(unpacked_sizes)-2:
            modules.append(nn.ReLU())
            if dropout >0.:
                modules.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*modules)
    return mlp

def setup_vocab(embedding_dim= 300):
    # with open(training_path) as f:
    #     data = json.load(f)
    # texts = [d['text'] for d in data]
    # tokenizer = get_tokenizer('basic_english')
    # tokenized_texts = [tokenizer(text) for text in texts]
    # Create a vocabulary from the tokenized texts
    myvec = GloVe(name='6B', dim=embedding_dim)
    # myvec = Vectors(name=f'glove/glove.6B.{embedding_dim}d.txt')
    pad_token = '<pad>'
    pad_token_id = 0
    myvocab = vocab(myvec.stoi)
    myvocab.insert_token(pad_token, pad_token_id)
    myvocab.set_default_index(pad_token_id)
    pretrained_embeddings = myvec.vectors
    pretrained_embeddings  = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
    return pretrained_embeddings,myvocab

class RobertaEncoderOnly(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.pooler = None  # remove pooler

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, 
                 bidirectional, dropout, pad_idx,vocab,pretrained_embeddings):
        super().__init__()
        
        # Load the GloVe embeddings
        self.vocab = vocab
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.embedding = nn.EmbeddingBag.from_pretrained(pretrained_embeddings,freeze=False)
        
        # Replace the random weights with the pre-trained GloVe weights
        # self.embedding.weight.data.copy_(self.vocab.vectors)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids,attention_mask = None):
        input_emb = self.dropout(self.embedding(input_ids))
        
        _, (out_hidden, _) = self.rnn(input_emb)
        
        if self.rnn.bidirectional:
            out_hidden = torch.cat((out_hidden[-2,:,:], out_hidden[-1,:,:]), dim = 1)
        else:
            out_hidden = out_hidden[-1,:,:]
        
        return self.dropout(out_hidden)


