import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import TokenClassificationDataset
import argparse
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification
)
from fairseq.optim.adafactor import Adafactor
# from configs.templates import *
# from configs.generate_config import *
from collections import defaultdict
from preprocess.data import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model.utils import *


class NLI_ETP(pl.LightningModule):
    def __init__(self, hparams):
        super(NLI_ETP, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.num_classes = hparams.dataset_info['num_classes']
        self.max_length = hparams.dataset_info['max_length'][hparams.model_name_or_path]
        self.model_name_keys = hparams.model_type.split('-')
        self.rationale_encoder = None
        if self.model_name_keys[0] == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)
            self.encoder_model = AutoModel.from_pretrained(hparams.model_name_or_path)
            hidden_dim = self.encoder_model.config.hidden_size
            self.predictor_ff = nn.Linear(
                    hidden_dim,
                    hparams.dataset_info['num_classes']
                )
            if hparams.plaus_weight > 0.:
                self.rationale_ff = nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(True),
                    nn.Dropout(p=hparams.dropout),
                    nn.Linear(hidden_dim, 1))
                
                if self.model_name_keys[-1] == 'dm':
                    self.rationale_encoder = AutoModel.from_pretrained(hparams.model_name_or_path)
            else:
                self.rationale_ff = None
            
        self.class_labels = {}
        
        ## load nli model
        print ("Loading pretrained NLI model")
        if hparams.nli_trained:
            add_str = 'trained'
        else:
            add_str = 'untrained'
        if hparams.answer_only:
            nli_path = f'nli_weights/{hparams.task}_nli_model_answer_{add_str}.pt'
        else:
            nli_path = f'nli_weights/{hparams.task}_nli_model_{int(hparams.pct_train_rationales*100)}_{add_str}.pt'

        self.nli_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-large')
        self.nli_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-large')
        if not hparams.unsupervised and not hparams.supervised:
            nli_weights = torch.load(nli_path,map_location = 'cpu')
            self.nli_model.load_state_dict(nli_weights)
        for param in self.nli_model.parameters(): # Set to False
            param.requires_grad = False
    def is_logger(self):
        return True
    
    def forward(
        self, batch,inference=False):
        shared_encoder = True if self.model_name_keys[-1] != 'dm' else False
        if not inference:
            loss_dict = {}
            if self.hparams.plaus_weight > 0.:
                sen_rationale_loss,masked_attention_mask,_,_ = get_logits_loss(batch,self.encoder_model,self.rationale_ff,self.tokenizer,self.nli_model,self.nli_tokenizer,args =self.hparams,shared_encoder = shared_encoder,separate_encoder = self.rationale_encoder,get_logits = False)   
                
                ## Task loss
                masked_attention_mask[:,0] = 1. # set cls to true
                inp_embeds = self.encoder_model.embeddings.word_embeddings(batch['input_ids'])
                if not self.hparams.soft:
                    inp_embeds = inp_embeds * masked_attention_mask.unsqueeze(-1) # if soft, just apply soft masking to attn mask
                out_states = self.encoder_model(inputs_embeds = inp_embeds,attention_mask = masked_attention_mask).pooler_output
                task_logits = self.predictor_ff(out_states)
            else:
                masked_attention_mask = batch['attention_mask']
                sen_rationale_loss = torch.tensor(0.0).to(self.device)
                masked_attention_mask[:,0] = 1. # set cls to true
                task_output = self.encoder_model(input_ids = batch['input_ids'],attention_mask = masked_attention_mask).pooler_output
                task_logits = self.predictor_ff(task_output)
                
            task_loss = nn.CrossEntropyLoss()(task_logits,batch['label'])
            
            loss = task_loss + self.hparams.plaus_weight * sen_rationale_loss
            loss_dict['task_loss'] = task_loss
            loss_dict['plaus_loss'] = sen_rationale_loss

            return loss,loss_dict
        else:
            test_dict = {}
            with torch.no_grad():
                if self.hparams.plaus_weight > 0.:
                    sent_r_probs,nli_logits,masked_attention_mask,selected_sen = get_logits_loss(batch,self.encoder_model,self.rationale_ff,self.tokenizer,self.nli_model,self.nli_tokenizer,args =self.hparams,shared_encoder = shared_encoder,separate_encoder = self.rationale_encoder,get_logits = True)
                
                else:
                    masked_attention_mask = batch['attention_mask']
                    sent_r_probs = None
                    nli_logits = None
                    
                masked_attention_mask[:,0] = 1. # set cls to true
                inp_embeds = self.encoder_model.embeddings.word_embeddings(batch['input_ids'])
                inp_embeds = inp_embeds * masked_attention_mask.unsqueeze(-1) 
                task_output = self.encoder_model(inputs_embeds = inp_embeds,attention_mask = masked_attention_mask).pooler_output
                task_logits = self.predictor_ff(task_output)
                y_hat_probs = torch.softmax(task_logits,dim = -1) 
                # get mask from selected sentences if enforce at least one.
                if sent_r_probs is not None:
                    if not self.hparams.soft:
                        if self.hparams.min_one: # select at least one sentence, use from selected_sen
                            nli_s_mask = torch.zeros_like(sent_r_probs).to(self.device)
                            for b,pos in enumerate(selected_sen):
                                nli_s_mask[b,pos] = 1.
                        else: # if not use threshold
                            nli_s_mask = (sent_r_probs > 0.5).float()
                    else: # soft mask
                        nli_s_mask = selected_sen # taken either with top p or top k
                else:
                    nli_s_mask = None
                # nli logits: (bs, max sen,no classes), sen_r_probs : (b,max sen)
                if self.hparams.align and nli_logits is not None and not self.hparams.supervised:
                    # nli_logits[:,:,2] = -1e4 # set neutral class to a large negative number (enforce selected sentence to only be pos or neg)
                    nli_probs = torch.softmax(nli_logits,dim = -1) # (b,max sen,no classes)
                    nli_p = nli_probs * nli_s_mask.unsqueeze(-1)
                    num_sens = torch.sum(nli_s_mask,dim = 1).unsqueeze(-1) # (b,1)
                    num_sens[num_sens == 0] = 1e9 # if there is no sentences selected, set to a large number to ignore later all when aligning
                    avg_p = torch.sum(nli_p,dim = 1) / num_sens # (b,no classes)

                    if self.hparams.task != 'evidence_inference':
                        y_hat_aligned = torch.max(avg_p[:,:2],y_hat_probs) # avg_p throw away neutral class and align
                    else:
                        y_hat_aligned = torch.max(avg_p,y_hat_probs) # has 3 classes
                    y_hat = torch.argmax(y_hat_aligned,dim = -1)
                else:
                    y_hat = torch.argmax(y_hat_probs,dim = -1)

                test_dict['z'] = nli_s_mask
                test_dict['z_probs'] = sent_r_probs
                test_dict['y_hat'] = y_hat
                
            return test_dict    

    def _step(self, batch,inference=False):
        if inference:
            test_dict = self(batch,inference = True) # contains z_logits, nli probs and task label
            test_dict['z_gold'] = batch['z_gold']
            test_dict['input_ids'] = batch['input_ids']
            test_dict['label'] = batch['label']
            test_dict['sen_mask'] = batch['sen_mask']
            test_dict['noisy_z'] = batch.get('noisy_z',None)
            return test_dict
        else:
            loss,loss_dict = self(batch,inference = False)
            return loss,loss_dict

    def training_step(self, batch, batch_idx):
        loss,train_loss_dict = self._step(batch)
        self.log("train_loss",loss,on_epoch=True,prog_bar=True,sync_dist=True if self.hparams.n_gpu > 1 else False) 
        for k,v in train_loss_dict.items():
            self.log(k,v,on_epoch=True,prog_bar=True,sync_dist=True if self.hparams.n_gpu > 1 else False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss,val_loss_dict = self._step(batch)
        self.log("val_loss",loss,on_epoch=True,prog_bar=True,sync_dist=True if self.hparams.n_gpu > 1 else False)
        self.log("epoch",int(self.current_epoch))
        for k,v in val_loss_dict.items():
            self.log(k,v,on_epoch=True,prog_bar=True,sync_dist=True if self.hparams.n_gpu > 1 else False)
        return {"val_loss": loss}

    # def on_validation_epoch_end(self):
    #     self.hparams.optimal_tok_threshold = np.mean(self.optimal_threshold['tok'])
    #     self.log("optimal_tok_threshold",self.hparams.optimal_tok_threshold,on_epoch=True,prog_bar=True,sync_dist=True if self.hparams.n_gpu > 1 else False)
    #     self.save_hyperparameters(self.hparams)
    #     self.optimal_threshold = defaultdict(list)
        
    
    def test_step(self, batch, batch_idx):
        test_result = self._step(batch,inference=True)
       
        test_results = get_metrics(test_result)
        
        for k,v in test_results.items():
            if v is not None:
                self.log(k,v,on_epoch=True,prog_bar=True,sync_dist=True if self.hparams.n_gpu > 1 else False)
        
    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model_dict= {}
        model_dict['enc_model'] = self.encoder_model
        model_dict['predictor_ff'] = self.predictor_ff
        if self.hparams.plaus_weight > 0.:
            model_dict['rationale_ff'] = self.rationale_ff
            if self.rationale_encoder is not None:
                model_dict['rationale_encoder'] = self.rationale_encoder
            
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_dict['enc_model'].named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model_dict['enc_model'].named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        for model_k,model_head in model_dict.items():
            if model_k != 'enc_model':
                optimizer_grouped_parameters += [
            {
                "params": [p for n, p in model_head.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model_head.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.hparams.optimizer == "adam":
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        elif self.hparams.optimizer == "adafactor":
            optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
            
        self.opt = optimizer
        return [optimizer]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure,optimizer_idx=None):
        
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()
    
    def train_dataloader(self):

        train_dataset = TokenClassificationDataset(self.tokenizer, self.hparams.data_dir,self.hparams.task,
                                                   split = 'train',
                                                   max_length = self.max_length,
                                                   nli_tokenizer = self.nli_tokenizer,
                                                   debug = self.hparams.debug,
                                                   answer_only=self.hparams.answer_only,
                                                   attack = self.hparams.attack,
                                                   pct_data = self.hparams.pct_supervised if self.hparams.supervised else 1.0)

        dataloader = DataLoader(train_dataset,
                    batch_size=self.hparams.train_batch_size,
                    collate_fn = train_dataset.collate_fn,
                    shuffle = True,
                    num_workers = 8,
                    pin_memory = False)
        self.total_size = len(train_dataset)
        
        t_total = (
            (self.total_size// (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        // self.hparams.gradient_accumulation_steps * float(self.hparams.num_train_epochs)
        )
        if self.hparams.warmup_steps >=1.0:
            warmup_steps = int(self.hparams.warmup_steps) 
        else:
            warmup_steps = int(self.total_size * self.hparams.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):

        val_dataset = TokenClassificationDataset(self.tokenizer, self.hparams.data_dir,self.hparams.task,
                                                   split = 'val',
                                                   max_length = self.max_length,
                                                   nli_tokenizer = self.nli_tokenizer,
                                                   debug = self.hparams.debug,
                                                   answer_only=self.hparams.answer_only,
                                                   attack = self.hparams.attack,
                                                   pct_data = self.hparams.pct_supervised if self.hparams.supervised else 1.0)
        val_dataloader = DataLoader(val_dataset,
                    batch_size=self.hparams.eval_batch_size,
                    collate_fn = val_dataset.collate_fn,
                    shuffle = False,
                    num_workers = 8,
                    pin_memory = False)
        return val_dataloader
    
    def test_dataloader(self):
        test_dataset = TokenClassificationDataset(self.tokenizer, self.hparams.data_dir,self.hparams.task,
                                                   split = 'test',
                                                   max_length = self.max_length,
                                                   nli_tokenizer = self.nli_tokenizer,
                                                   debug = self.hparams.debug,
                                                   answer_only=self.hparams.answer_only,
                                                   attack = self.hparams.attack)

        test_dataloader = DataLoader(test_dataset,
                    batch_size=self.hparams.test_batch_size,
                    collate_fn = test_dataset.collate_fn,
                    shuffle = False,
                    num_workers = 8,
                    pin_memory = False)
        return test_dataloader



class LoggingCallback(pl.Callback):
    def __init__(self,kfold=0,task="classification",expl_label='f',file_path=None,args=None):
        self.kfold = kfold
        self.task = task
        self.expl_label = expl_label
        self.args = args
        self.file_path = file_path
        
    def on_validation_end(self, trainer, pl_module):
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        if metrics['epoch'] == 0:
            writer_mode = "w"
        else:
            writer_mode = "a"
        file_path= self.file_path # set to movie as ood dataset only used for test
        
        if trainer.global_rank == 0:    
            with open(file_path, writer_mode) as writer:
                if writer_mode == "w":
                    writer.write("***** Validation results *****\n")
                writer.write(f"Epoch: {metrics['epoch']}\n")
                for key in sorted(metrics):
                    if key !="epoch":
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        test_metrics = trainer.callback_metrics
        if trainer.global_rank == 0:
            with open(self.file_path,'a',encoding = 'utf-8') as f:
                for k,v in test_metrics.items():
                    f.write(f"{k}: {str(np.round(v.cpu().numpy(),3))}\n")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str, required=False, default = 'data')
    parser.add_argument("--output_dir", type=str, required=False, default="results")
    parser.add_argument("--model_name_or_path",type=str, required=False, default='roberta-base')
    parser.add_argument("--tokenizer_name_or_path", type=str, required=False, default='roberta-base')
    parser.add_argument("--model_type", type=str, required=False, default='roberta-sm',help = 'sm = shared model, dm = different model')
    parser.add_argument("--supervised", type=bool, required=False, default=False,help = 'if to use gold z')
    parser.add_argument("--unsupervised", type=bool, required=False, default=False,help = 'without fine-tuning the nli model')
    parser.add_argument("--align", type=bool, required=False, default=False,help = 'to align task label to nli prediction, inference stage')
    parser.add_argument("--min_one", type=bool, required=False, default=False,help = 'select at least one sen')
    parser.add_argument("--learning_rate",  type=float, required=False, default=2e-5)
    parser.add_argument("--adam_epsilon",  type=float, required=False, default=1e-8)
    parser.add_argument("--warmup_steps", type=float, required=False, default=0.1)
    parser.add_argument("--train_batch_size",  type=int, required=False, default=8)
    parser.add_argument("--eval_batch_size",  type=int, required=False, default=8)
    parser.add_argument("--test_batch_size",  type=int, required=False, default=8)
    parser.add_argument("--num_train_epochs", type=int, required=False, default=10)
    parser.add_argument("--dropout", type=float, required=False, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=False, default=1)
    parser.add_argument("--fp_16",  type=bool, required=False, default=True,help = "if you want to enable 16-bit training then install apex and set this to true")
    parser.add_argument("--max_grad_norm",  type=float, required=False, default=1.0)
    parser.add_argument("--seed",type=int, required=False, default=42)
    parser.add_argument("--gpu_no", nargs = '+',type=int, required=False, default=1)
    parser.add_argument("--tasks", nargs ='+', type=str, required=False, default='fever') #  a list of strs
    parser.add_argument("--weight_decay",  type=float, required=False, default=0.0)
    parser.add_argument("--plaus_weight",  type=float, required=False, default=1.0)
    parser.add_argument("--strategy",  type=str, required=False, default='ddp') # how many few-shot examples
    parser.add_argument("--optimizer",  type=str, required=False, default='adam')
    parser.add_argument("--load_model",  type=bool, required=False, default=False)
    parser.add_argument("--soft",  type=bool, required=False, default=False) # if to use gumbel-softmax to ensure differientability
    parser.add_argument("--tau",  type=float, required=False, default=0.5)
    parser.add_argument("--debug",  type=bool, required=False, default=False) # set num data to only 100 points
    parser.add_argument("--answer_only",  type=bool, required=False, default=False)
    parser.add_argument("--evaluate_robustness",  type=bool, required=False, default=False) # if to evalaute robustness, should have attack files in data dir
    parser.add_argument("--pct_train_rationales",  type=float, required=False, default=0.1)
    parser.add_argument("--pct_supervised",  type=float, required=False, default=1.0,help= 'Only for Supervised, to compare between 10\% and 10% rationale')
    parser.add_argument("--nli_trained",  type=bool, required=False, default=False,help = 'whether to use a nli trained model')
    args = parser.parse_args()
    set_seed(args.seed)

    args.n_gpu = len(args.gpu_no)
    
    n_devices = torch.cuda.device_count()
    if args.n_gpu > n_devices:
        args.n_gpu = n_devices
    
    if args.n_gpu <= 1:
        args.strategy = 'auto'
    
    if not os.path.exists(args.output_dir):
        os.mkdir(path=args.output_dir)

    if '/' in args.model_name_or_path:
        model_name = args.model_name_or_path.split('/')[-1]
    else:
        model_name = args.model_name_or_path
    
    ##  Setting up results files
    if args.plaus_weight <= 0.:
        model_name += '_no_plaus'
    elif args.supervised:
        if args.pct_supervised < 1.0:
            model_name += f'_S_{int(args.pct_supervised*100)}'
        else:
            model_name += '_S'
    else:
        model_name += ('-' + args.model_type.split('-')[-1])
        model_name += f'_{int(args.pct_train_rationales*100)}'
    if args.min_one:
        model_name += '_mo'
    if args.answer_only:
        model_name += '_ao'
    if args.unsupervised:
        model_name += '_US'
    
    
    
    if args.soft:
        if args.min_one and model_name.endswith('_mo'):
            model_name = model_name.split('_mo')[0]
        model_name += '_soft'

    all_tasks = args.tasks
    output_dir = os.path.join(args.output_dir,'single')
    output_num_dir = os.path.join(output_dir,'num')
    makefile(output_num_dir)
    output_num_file = defaultdict()
    
    if args.nli_trained:
        model_name += '_trained'
    else:
        model_name += '_untrained'
    
    for task in all_tasks:
        num_taskdir = os.path.join(output_num_dir,task,str(args.seed))
        makefile(num_taskdir)
        output_num_file[task] = os.path.join(num_taskdir,f"{model_name}_{args.plaus_weight}.txt") 
        if args.align:
            output_num_file[task] = output_num_file[task].split('.txt')[0] + '_align.txt'

    if not os.path.exists(output_dir):
        os.mkdir(path=output_dir)
    
    output_text_dir = os.path.join(output_dir,'text')
    makefile(output_text_dir)
    
    output_text_file = defaultdict()
    for task in all_tasks:
        task_dir = os.path.join(output_text_dir,task,str(args.seed))
        makefile(task_dir)
        output_text_file[task] = os.path.join(task_dir,f"{model_name}_{args.plaus_weight}.txt") 
        if args.align:
            output_text_file[task] = output_text_file[task].split('.txt')[0] + '_align.txt'
 
    ## define training params
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=False,
        mode='min')
    
    callbacks = [LoggingCallback(task = args.tasks,file_path = output_num_file,args=args)
                 ,early_stop_callback
                #  ,checkpoint_callback
                 ]
    print (f"running gpus: {args.gpu_no}")
    train_params = dict(
                        strategy =  args.strategy,
                        accumulate_grad_batches=args.gradient_accumulation_steps,
                        accelerator='gpu' if args.n_gpu > 0 else None,
                        devices = args.gpu_no,
                        max_epochs=args.num_train_epochs,
                        precision=16 if args.fp_16 else 32,
                        gradient_clip_val=args.max_grad_norm, # default args.max_grad_norm
                        enable_checkpointing =True,
                        callbacks=callbacks,
                        # log_every_n_steps = sum([len(t[0]) for t in curr_train_ids])//args.train_batch_size
                        )
    
    model_ckpt_dir = 'model_checkpoints'

    ## initalize model
    for task in all_tasks:
        print("Training for task: ",task)
        args.task = task
        args.text_file = output_text_file[task]
        
        args.attack = False # important to set to false for test for the 1st run
        
        ds_info = dataset_info[task]
        args.dataset_info = ds_info
        args.top_k = ds_info['top_k'] # set for soft approach
        args.top_p = ds_info['top_p']
        
        
        # with open(args.text_file,'w') as fp:
        #     fp.write('Noisy text logging\n\n')
        
        # if task in dataset_batch_size.keys():
        #     args.train_batch_size = dataset_batch_size[task]
        #     args.eval_batch_size = dataset_batch_size[task]
        #     args.test_batch_size = dataset_batch_size[task]
            
        # if args.model_name_or_path.split('-')[-1] == 'large' and args.train_batch_size >= 4:
        #     args.train_batch_size = int(args.train_batch_size//4)
        #     args.eval_batch_size = int(args.eval_batch_size//4)
        #     args.test_batch_size = int(args.test_batch_size//4)
        
        ## saved model name
        model_filename = f"{model_name}_{task}_{args.plaus_weight}_{str(args.seed)}_{int(args.pct_train_rationales*100)}"

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_ckpt_dir,
        filename=model_filename,  # your custom filename
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        overwrite_existing=True
        )

        ## batch size from config file
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=False,
            mode='min')
        
        ## CALLBACKS
        callbacks = [LoggingCallback(task = task,file_path = output_num_file[task],args=args)
                ,early_stop_callback
                ,checkpoint_callback
                ]
        train_params['callbacks'] = callbacks
        trainer = pl.Trainer(**train_params)
        if args.load_model:
            with open(output_num_file[task], 'a') as fp:
                fp.write('Test result for loaded Model!\n\n')
                fp.write("="*100+'\n')
            
            
            model = NLI_ETP.load_from_checkpoint(checkpoint_path = os.path.join(model_ckpt_dir,f'{model_filename}.ckpt'),hparams=args,map_location = 'cpu')
            test_metrics = trainer.test(model)
            if args.evaluate_robustness:
                args.attack = True # set to true to change the dir of the data loader and reload the model
                model = NLI_ETP.load_from_checkpoint(checkpoint_path = os.path.join(model_ckpt_dir,f'{model_filename}.ckpt'),hparams=args,map_location = 'cpu')
                _ = trainer.test(model)
                diff_task,diff_plaus = get_metric_difference(output_num_file[task])
                with open(output_num_file[task], 'a') as fp:
                    fp.write('Diff task f1: {:.2f}\n'.format(diff_task))
                    fp.write('Diff token f1: {:.2f}\n'.format(diff_plaus))
                
            
        else:
            args.optimal_tok_threshold = 0.5
            model = NLI_ETP(args)
            trainer.fit(model)
            # test_metrics = trainer.test(model,
            #                             ckpt_path='best'
            #                             )
        # test_metrics = test_metrics[0]
        # if trainer.global_rank == 0:
        #     with open(output_num_file[task],'a',encoding = 'utf-8') as f:
        #         f.write("="*100+'\n')
        #         for k,v in test_metrics.items():
        #             f.write(f"{k}: {str(np.round(v,3))}\n")
            print ("FINISHED TRAINING FOR TASK: ",task)

if __name__ == "__main__":
    try:
        main()
        
    except KeyboardInterrupt:
        print("Interrupted by user")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
