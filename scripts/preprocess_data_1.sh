export PYTHONPATH=$PYTHONPATH:/~/finbert/finBERT/t5  # set to root dir
export CUDA_VISIBLE_DEVICES=6
batchsize=32
dataset=boolq

#arguements - dataset: dataset, arch: tokenizer model name, data_dir: dir of eraser data pct_train_rationales = amount of supervised z
for supervised_percent in 0.1 0.25
do
    ## To get nli data for to train f_nli
    python preprocess/get_nli_data.py --dataset $dataset --pct_train_rationales $supervised_percent 

    # # to train the f_nli model:
    python train_nli.py --dataset_name $dataset --pct_train_rationales $supervised_percent --batch_size $batchsize

    # # to preprocess eraser data to train ETP model
    python preprocess/preprocess_eraser.py --dataset $dataset 
done

