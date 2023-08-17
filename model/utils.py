import torch
import os
import numpy as np
from typing import List,Tuple
from sklearn.metrics import f1_score,accuracy_score,precision_recall_curve,auc,roc_curve,recall_score,precision_score,recall_score
import torch.nn as nn
from sklearn.metrics.pairwise import linear_kernel
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import math
import random

# A function to get the start and end position of spans
def get_spans(arr: np.ndarray) -> List[Tuple[int, int]]:
    # Find the start and end of each span
    spans = []
    start = None
    for i, x in enumerate(arr):
        if x == 1 and start is None:
            start = i
        elif x == 0 and start is not None:
            spans.append((start, i))
            start = None
    # Handle the case where the array ends with 1
    if start is not None:
        spans.append((start, len(arr)))
    return spans

def compute_iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    start1, end1 = span1
    start2, end2 = span2
    start_max = max(start1, start2)
    end_min = min(end1, end2)
    intersection = max(0, end_min - start_max)
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union > 0 else 0.

def compute_iou_f1(y_true,y_pred,threshold:float = 0.5):
    pred_spans = get_spans(y_pred)
    true_spans = get_spans(y_true)
    best_iou_scores,pred_binary = [],[]
    
    for true_span in true_spans:
        best_iou = 0.
        for pred_span in pred_spans:
            iou_score = compute_iou(true_span,pred_span)
            best_iou = max(best_iou,iou_score)
        best_iou_scores.append(best_iou)
        if best_iou > threshold:
            pred_binary.append(1)
        else:
            pred_binary.append(0)
    iou_f1 = sum(pred_binary)/len(pred_binary) if len(pred_binary) > 0 else 0.

    return iou_f1,np.mean(best_iou_scores)if len(best_iou_scores) > 0 else 0.

def get_threshold(probs,labels):
    probs = probs.float().flatten().cpu().detach().numpy()
    labels = labels.flatten().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels, probs)
    optimal_idx = np.argmax(tpr - fpr)
    sen_optimal_threshold = thresholds[optimal_idx]
    return sen_optimal_threshold

def preprocess_query(query,task,answer_only=False):
    if task == 'boolq':
        return query.strip() + '.'
    elif task == 'multirc':
        query = query.split('||')
        q = query[0].strip()
        a = query[1].strip()
        if answer_only:
            return a + '.'
        else:
            return q + ' ' + a + '.'
    elif task == 'evidence_inference':
        query = query.split('|')
        treatment = query[0].strip()
        compare = query[1].strip()
        outcome = query[2].strip()
        out = treatment + ' compared to ' + compare + ' for ' + outcome + ' is increased.'
        return out
    else:
        return query.strip()

"""
Adapted from https://github.com/bhargaviparanjape/explainable_qa.git,
for boolq and evidence-inference dataset, we use tf-idf to select the most relevant document
"""
def tfidf_selection(tfidf_vectorizer,sentences,query,evidence_spans,max_sen = 20):
    ## evidence spans have both start and end, since boolq and evidence-inference have multiple spans
    ## sentences is a list of lists of tokens
    all_docs = [query]
    sent_spans = []
    for span_start in range(0, len(sentences) - max_sen + 1, 5):
        sentences_in_span = sentences[span_start:span_start + max_sen]
        paragraph = [tok for sentence in sentences_in_span for tok in sentence]
        sent_spans.append((span_start, span_start + max_sen))
        all_docs.append(" ".join(paragraph))
    last_paragraph = [tok for sentence in sentences[-max_sen:] for tok in sentence]
    if len(sentences) - max_sen > 0:
        all_docs.append(" ".join(last_paragraph))
        sent_spans.append((len(sentences) - max_sen, len(sentences)))
    else:
        # doc has fewer than max_num_sentences
        all_docs.append(" ".join(last_paragraph))
        sent_spans.append((0, len(sentences)))
    tfidf_vecs = tfidf_vectorizer.fit_transform(all_docs)
    cosine_similarities = linear_kernel(tfidf_vecs[0:1], tfidf_vecs).flatten()
    evidence_in_span = False
    out_evidence_spans = []
    for i in range(0,len(cosine_similarities[1:])): # skip the first one since it is the query
        best_window = np.argsort(cosine_similarities[1:])[::-1][i]
        best_sentence_span = (sent_spans[best_window][0],sent_spans[best_window][1])
        for evidence_span in evidence_spans:
            if evidence_span[0] >= best_sentence_span[0] and evidence_span[1] <= best_sentence_span[1]:
                evidence_in_span = True
                out_evidence_spans.append((evidence_span[0] - best_sentence_span[0], evidence_span[1] - best_sentence_span[0]))
        if evidence_in_span: # once have evidence break, else continue down the tf-idf list
            sentences = sentences[sent_spans[best_window][0]:sent_spans[best_window][1]]
            out_sentences = [" ".join(s) for s in sentences]
            break
    if not evidence_in_span:
        return None,None
    return out_sentences,out_evidence_spans
            
    

def get_logits_loss(batch,sent_encoder,sent_head,tokenizer,nli_model,nli_tokenizer,args=None,shared_encoder=False,separate_encoder=None,get_logits=False):
    token_ids = batch['input_ids']
    token_mask = batch['attention_mask']
    sen_start = batch['sen_start']
    sen_end = batch['sen_end']
    sen_mask = batch['sen_mask'] # (batch size, max sentences)
    nli_inputs = batch['nli_inputs'] # (num batch,max sentences, max sentence length)
    nli_mask = batch['nli_mask'] # (num batch,max sentences, max sentence length)
    label = batch['label']
    
    batch_size = token_ids.shape[0]
    
    if shared_encoder:  # share encoder
        e_out = sent_encoder(input_ids = token_ids,attention_mask = token_mask)
    else:
        e_out = separate_encoder(input_ids = token_ids,attention_mask = token_mask)
        
    e_states = e_out.last_hidden_state
    
    ## get sentence-level loss from self-supervised label
    selected_mask = torch.zeros_like(token_mask)
    

    sen_states  = torch.cat([e_states.gather(dim=1,index = sen_start.unsqueeze(-1).expand(-1,-1,e_states.shape[-1])),e_states.gather(dim=1,index = sen_end.unsqueeze(-1).expand(-1,-1,e_states.shape[-1]))],dim=-1)
    sen_logits = sent_head(sen_states).squeeze(-1)
    sen_probs = torch.sigmoid(sen_logits) * sen_mask # mask out padded sentences

    if not args.supervised:
        with torch.no_grad():
            max_sen_length = nli_inputs.shape[-1]
            nli_logits = nli_model(input_ids = nli_inputs.reshape(-1,max_sen_length),attention_mask = nli_mask.reshape(-1,max_sen_length)).logits
            out_nli_logits = nli_logits.reshape(batch_size,-1,nli_logits.shape[-1])
            ## to get self-supervised rationale labels only during training
            if not get_logits:
                nli_probs = torch.softmax(nli_logits,dim=-1) * sen_mask.reshape(-1).unsqueeze(-1).expand(-1,nli_logits.shape[-1]) # clear the probs of padding sentences
                r_label = get_nli_label(nli_probs,label,sen_mask,soft=args.soft) 
            else:
                r_label = None

        ## if hard, set mask according to hard labels
        if not args.soft:       
            selected_mask,selected_sen = set_mask(sen_probs,sen_start,sen_end,selected_mask,args.min_one)
        else: # soft approach use gumbel-softmax with reparameterization while weighing the explainer logits with nli feedback
            if not get_logits:
                weighted_probs = sen_probs * r_label[:,:,1] # weight the probs of explainer with the nli feedback directly
                weighted_logits = torch.log((weighted_probs/(1.-weighted_probs))+1e-7) # transform back to logits for gumbel-sofmax
                sen_dist = nn.functional.logsigmoid(weighted_logits)
                soft_probs = reparameterize(sen_dist,tau=args.tau)
                selected_mask = set_mask_soft(soft_probs,sen_start,sen_end,max_token_length = token_mask.shape[1]) # get soft mask, have to maintain differientability
            else: # get a hard mask via top p or top k, do not use nli feedback at all.
                selected_mask,selected_sen = set_mask_hard(sen_probs,selected_mask,sen_start,sen_end,args.top_p,args.top_k)
                    
    else:
        if not get_logits:
            r_label = batch['z_gold']
            selected_mask,_ = set_mask(r_label,sen_start,sen_end,selected_mask,args.min_one) # use gold z to get selected mask
        else:
            selected_mask,selected_sen = set_mask(sen_probs,sen_start,sen_end,selected_mask,args.min_one)
            out_nli_logits = None
        
    if not get_logits: # for training
        if not args.soft:
            sen_rationale_loss = (nn.functional.binary_cross_entropy_with_logits(sen_logits,r_label.float(),reduction='none')*sen_mask).sum()/sen_mask.sum()
        else:
            sen_rationale_loss = torch.tensor([0.]).to(sen_logits.device)

        return sen_rationale_loss,selected_mask,None,None
    else: # during inference time
        return sen_probs,out_nli_logits,selected_mask,selected_sen

def reparameterize(probs,tau=1.): # probs = (batch size, max sentences)
    probs = probs.unsqueeze(-1).view(probs.shape[0],1,-1)
    b_dist = RelaxedBernoulli(tau, logits=probs)
    z_out = b_dist.rsample().squeeze(1)
    return z_out
    
    


def get_nli_label(nli_probs,label,sen_mask,soft=False): 
    """
    mapping from ['contradiction', 'entailment', 'neutral'] to [True, False] via label [false, true]
    nli_logits: [batch_size,3]
    label: batch size
    nli_logits: (batch size * max num sentences,3)
    sen_mask: (batch size, max num sentences) # use this to index and take all the non padded outputs
    
    """
    ## shapes
    batch_size = label.shape[0]
    max_sentences = sen_mask.shape[-1]
    
    nli_probs = nli_probs.reshape(batch_size,max_sentences,-1)
    argmax_values = torch.argmax(nli_probs, dim=-1)
    out_binary_labels = []
    ## Gather the labels and preds for each sentence and change to binary labels
    for i in range(argmax_values.shape[0]):
        curr_len = sen_mask[i].sum().item()
        gathered_labels = label[i].repeat(curr_len)
        if not soft:
            gathered_preds = argmax_values[i,:curr_len]
            out_labels = torch.zeros(curr_len)
            # Update output where list1 is 0 and list2 is 0
            out_labels[(gathered_preds == 0) & (gathered_labels == 0)] = 1
            # Update output where list1 is 1 and list2 is 1
            out_labels[(gathered_preds == 1) & (gathered_labels == 1)] = 1

            ## Handle the cases where there is no selected sentence
            if torch.sum(out_labels) <1: ## train on large quantity of neturals
                out_labels = get_label_from_probs(nli_probs[i],label[i],curr_len)
            ## pad back
            out_binary_labels.append(torch.nn.functional.pad(out_labels.long(),(0,max_sentences-curr_len),value = 0.))
           
        else:
            out_labels = nli_probs[i,:curr_len].clone()

            # sums_0 = out_labels[:,1] + out_labels[:,2]
            # sums_1 = out_labels[:,0] + out_labels[:,2]
            sums = out_labels[:,2]

            out_labels[gathered_labels==0,1] = out_labels[gathered_labels==0,0] # shift the contradiction to 2nd column of selected
            out_labels[gathered_labels==0,0] = sums[gathered_labels==0] # set the 0 to the 1st column
            out_labels[gathered_labels==1,0] = sums[gathered_labels==1] # set the entailment to 2nd column of selected


            out_labels = torch.cat([out_labels[:,:2],torch.zeros((max_sentences-curr_len,2)).to(out_labels.device)],dim=0)
            out_binary_labels.append(out_labels)
         
    out_binary_labels = torch.stack(out_binary_labels,dim=0).to(nli_probs.device)
    
    return out_binary_labels


def get_label_from_probs(probs,label,len):
    """
    handle cases where there is no selected sentence
    """
    probs = probs[:len] # take until the length of the sentence (num sentence, 3)
    out_labels = torch.zeros(len)
    if label.long() == 1: ## True, find max entail scores
        top_sen_id = torch.argmax(probs[:,1])
    else: ## False, find max contradiction scores
        top_sen_id = torch.argmax(probs[:,0])
    out_labels[top_sen_id] = 1
    return out_labels

def set_mask(probs,sen_start,sen_end,mask,min_one=False):
    condition = probs > 0.5
    sen_mask = []
    ## index to create a mask from selected sentences
    for batch_idx in range(condition.size(0)):
    # Identify the sentences with sen_probs > 0.5 ## possible that there is no selected sentence
        selected_sentences = condition[batch_idx].nonzero().squeeze(-1)   
        if selected_sentences.shape[0] == 0 and min_one: # no selected sentences and enforce at least 1
            selected_sentences = torch.argmax(probs[batch_idx]).unsqueeze(-1) # choose best sentence
        ## Always set query to be selected along with CLS token
        query_len = sen_start[batch_idx][0].item()
        mask[batch_idx,:query_len] = 1
        if selected_sentences.shape[0] > 0: # if there is at least 1 selected sentence
            for sentence_idx in selected_sentences:
                start = sen_start[batch_idx][sentence_idx].item()
                end = sen_end[batch_idx][sentence_idx].item()
                mask[batch_idx][start:end] = 1
        
        ## store the selected sentence pos
        sen_mask.append(selected_sentences)
    return mask,sen_mask

def set_mask_soft(probs,sen_start,sen_end,max_token_length = 512,eps = 1e-16):
    token_mask = []
    for b in range(probs.shape[0]):
        query_len = sen_start[b][0].item()
        mask = [torch.ones((query_len)).to(probs.device)] # query set to 1.
        for sen_idx in range(probs.shape[1]):
            start = sen_start[b][sen_idx].item()
            end = sen_end[b][sen_idx].item()
            if start == 0: # is padded
                break
            mask.append(probs[b][sen_idx].expand(end-start))
        mask = torch.cat(mask,dim=0) + eps # to prevent numerical instability
        mask = nn.functional.pad(mask,(0,max_token_length - mask.shape[0]),value = 0.)
        token_mask.append(mask)
    return torch.stack(token_mask,dim=0).to(probs.device)

def set_mask_hard(probs,mask,sen_start,sen_end,top_p=0.,top_k=1):
    """
    probs is of shape ( batch size, num sentences)
    mask is token-level shape (batch size, num tokens)
    
    return
    out_sen_mask = (batch size, num sentences), 1 if selected, 0 if not selected
    mask = (batch size, num tokens), 1 if selected, 0 if not selected
    """
    out_sen_mask = torch.zeros_like(probs).to(probs.device) # sentence prob
    for b in range(probs.shape[0]):
        query_len = sen_start[b][0].item()
        mask[b,:query_len] = 1. # query set to 1.
        if top_p > 0.: # we pick top p% of sentences
            no_sen = torch.nonzero(sen_start[b]).size(0)
            pick_sen = max(1,math.ceil(top_p * no_sen))
        else:
            pick_sen = top_k
        top_idx = torch.argsort(probs[b],descending=True)[:pick_sen]
        
        out_sen_mask[b,top_idx] = 1
        for idx in top_idx:
            start,end = sen_start[b][idx].item(),sen_end[b][idx].item()
            mask[b,start:end] = 1.
    return mask,out_sen_mask
        
            
        
def get_metrics(test_dict):
    results = {}
    ## task scores
    inp_ids = test_dict['input_ids']
    y_hat = test_dict['y_hat']
    y = test_dict['label']
    z_gold = test_dict['z_gold']
    z = test_dict['z']
    z_probs = test_dict['z_probs']
    z_mask = test_dict['sen_mask']
    noisy_z = test_dict.get('noisy_z',None)

    task_accuracy = accuracy_score(y.cpu(),y_hat.cpu())
    task_f1 = f1_score(y.cpu(),y_hat.cpu(),average='macro') 

    if z is not None:
        ## plausibility scores
        batch_size = z.shape[0]
        token_f1,auprc_score,total_recall,total_precision,AR = [],[],[],[],[]
        pred_ones,true_ones = [],[]
        for i in range(batch_size):
            num_sen = z_mask[i].sum()
            if torch.sum(z_gold[i][:num_sen]) > 0:
                curr_z_hat = z[i][:num_sen].long()
                curr_z_probs = z_probs[i][:num_sen]
                curr_z_gold = z_gold[i][:num_sen].long()
                if noisy_z is not None:
                    curr_noisy_z = noisy_z[i][:num_sen].long()
                    AR.append(recall_score(curr_noisy_z.cpu().numpy(),curr_z_hat.cpu().numpy(),zero_division= 0)) # attack capture rate of noisy z
                    
                ## for analysis of BoolQ
                pred_ones.append(torch.sum(curr_z_hat).item())
                true_ones.append(torch.sum(curr_z_gold).item())
                
                f1_s = f1_score(curr_z_gold.cpu().numpy(),curr_z_hat.cpu().numpy(),zero_division= 0)
                recall_s = recall_score(curr_z_gold.cpu().numpy(),curr_z_hat.cpu().numpy(),zero_division= 0)
                precision_s = precision_score(curr_z_gold.cpu().numpy(),curr_z_hat.cpu().numpy(),zero_division= 0)
                token_f1.append(f1_s)
                total_recall.append(recall_s)
                total_precision.append(precision_s)
                ## auprc
                precision,recall,_ = precision_recall_curve(y_true = curr_z_gold.cpu().numpy(),probas_pred= curr_z_probs.cpu().numpy())
                auprc_score.append(auc(recall,precision))

        avg_token_f1 = np.mean(token_f1)
        avg_auprc_score = np.mean(auprc_score)
        avg_recall = np.mean(total_recall)
        avg_precision = np.mean(total_precision)
        avg_pred_ones = np.mean(pred_ones)
        avg_true_ones = np.mean(true_ones)
        if len(AR) > 0:
            avg_AR = np.mean(AR)
        else:
            avg_AR = None
    else:
        avg_token_f1,avg_auprc_score,avg_recall,avg_precision,avg_pred_ones,avg_true_ones,avg_AR = None,None,None,None,None,None,None
        
    if noisy_z is not None: # only need these
        results['noisy_task_f1'] = task_f1
        results['noisy_token_f1'] = avg_token_f1
        results['AR'] = avg_AR
    else:
        results['task_acc'] = task_accuracy
        results['task_f1'] = task_f1
        results['token_f1'] = avg_token_f1
        results['recall'] = avg_recall
        results['precision'] = avg_precision
        results['auprc'] = avg_auprc_score
    if avg_pred_ones is not None:
        results['num_ones'] = avg_true_ones/avg_pred_ones
    
    return results

def get_metric_difference(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
    for line in lines[::-1]:
        if line.startswith('task_f1'):
            orignal_f1 = float(line.split(':')[-1].strip())
        elif line.startswith('noisy_task_f1'):
            noisy_f1 = float(line.split(':')[-1].strip())
        elif line.startswith('token_f1'):
            orignal_token_f1 = float(line.split(':')[-1].strip())
        elif line.startswith('noisy_token_f1'):
            noisy_token_f1 = float(line.split(':')[-1].strip())
        elif line.startswith('=='): # end of results
            break
    diff_task_f1 = ((orignal_f1-noisy_f1)/orignal_f1)*100
    try:
        diff_token_f1 = ((orignal_token_f1-noisy_token_f1)/orignal_token_f1)*100
    except:
        diff_token_f1 = None
    return diff_task_f1,diff_token_f1
            
def makefile(file_path):
    if not os.path.exists(file_path):
        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)