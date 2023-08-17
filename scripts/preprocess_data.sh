export PYTHONPATH=$PYTHONPATH:/~/finbert/finBERT/t5  # set to root dir

#arguements - dataset: dataset, arch: tokenizer model name, data_dir: dir of eraser data pct_train_rationales = amount of supervised z

## To get nli data for to train f_nli
python preprocess/get_nli_data.py --dataset multirc --pct_train_rationales 0.1

# # to train the f_nli model:
python train_nli.py --dataset_name multirc --pct_train_rationales 0.25 --device cuda:3

## to preprocess eraser data to train ETP model
python preprocess/preprocess_eraser.py --dataset multirc 
