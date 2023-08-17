import os
import numpy as np

# Change accordingly.
dataset = 'boolq' # dataset
data_path = 'results/single/num'
ao_only = False
us_pct = '10' # how much pct of supervised data used for NLI predictor. 

new_data_path = f'results/final/{dataset}'
if not os.path.exists(new_data_path):
    os.makedirs(new_data_path)
new_datafile = os.path.join(new_data_path, 'combined.txt')
dataset_path = os.path.join(data_path, dataset)

def get_filename(filename,file_type):
    if file_type == 'no_plaus':
        filename += '_no_plaus_0.0.txt'
        write_name = 'Full Context'
    elif file_type == 'supervised':
        filename += '_S_1.0.txt'
        write_name = 'Supervised (GOLD)'
    elif file_type == 'hard':
        filename += f'-sm_{us_pct}_mo_1.0.txt'
        write_name = 'Hard '
    elif file_type == 'hard_align':
        filename += f'-sm_{us_pct}_mo_1.0_align.txt'
        write_name = 'Hard with align'
    elif file_type == 'soft':
        filename += f'-sm_{us_pct}_soft_1.0.txt'
        write_name = 'Soft '
    elif file_type == 'soft_align':
        filename += f'-sm_{us_pct}_soft_1.0_align.txt'
        write_name = 'Soft with align'
    return filename,write_name
    

# every approach
for i,file_type in enumerate(['supervised']):
    task_acc = []
    task_f1 = []
    token_f1 = []
    precision = []
    recall = []
    filename = 'roberta-base' # change if using different model
    filename,write_name = get_filename(filename,file_type)
    ## every seed
    for seed in range(42,45):
        seed_path = os.path.join(dataset_path, str(seed))
        with open(os.path.join(seed_path,filename),'r') as f:
            lines = f.readlines()
            for line in lines[::-1]:
                if line.startswith('task_acc'):
                    task_acc.append(float(line.split(':')[-1].strip()))
                elif line.startswith('task_f1'):
                    task_f1.append(float(line.split(':')[-1].strip()))
                elif line.startswith('token_f1'):
                    token_f1.append(float(line.split(':')[-1].strip()))
                elif line.startswith('recall'):
                    recall.append(float(line.split(':')[-1].strip()))
                elif line.startswith('precision'):
                    precision.append(float(line.split(':')[-1].strip()))
                elif line.startswith('=='): # end of results
                    break
                
    with open(new_datafile,'a') as f:
        f.write(write_name+'\n')
        f.write('='*90+'\n')
        f.write('task_acc: '+str(np.round(np.mean(task_acc),3))+'\n')
        f.write('task_f1: '+str(np.round(np.mean(task_f1),3))+'\n')
        if len(token_f1) > 0:
            f.write('token_f1: '+str(np.round(np.mean(token_f1),3))+'\n')
        if len(precision) > 0:
            f.write('precision: '+str(np.round(np.mean(precision),3))+'\n')
        if len(recall) > 0:
            f.write('recall: '+str(np.round(np.mean(recall),3))+'\n')
        f.write('='*90+'\n')
        