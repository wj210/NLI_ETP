
export OMP_NUM_THREADS=8 
model_name='roberta-base'
batchsize=24
export CUDA_VISIBLE_DEVICES=0,4
ds=fever
nli_trained=True

for seed in 42 
do
    # blackbox model , set plaus_weight to 0
    # python main.py --gpu_no 1 --tasks $ds --plaus_weight 0.0 --seed $seed  
    # python main.py --gpu_no 1 --tasks $ds --plaus_weight 0.0 --seed $seed --load_model True

    # # blackbox model , set supervised to True, plaus to 1.0
    torchrun --nproc_per_node=2 main.py --gpu_no 0 1 --tasks $ds --plaus_weight 1.0 --seed $seed --supervised True --pct_supervised 0.1
    torchrun --nproc_per_node=2 main.py --gpu_no 0 1 --tasks $ds --plaus_weight 1.0 --seed $seed --supervised True --load_model True --pct_supervised 0.1

    ## Ours (hard) 
    # python main.py --gpu_no 0 --tasks $ds --plaus_weight 1.0 --seed $seed --min_one True --model_name_or_path $model_name --tokenizer_name_or_path $model_name --train_batch_size $batchsize --eval_batch_size $batchsize --pct_train_rationales $pct_train_rationales --nli_trained $nli_trained
    # python main.py --gpu_no 0 --tasks $ds --plaus_weight 1.0 --seed $seed --min_one True --load_model True  --model_name_or_path $model_name --tokenizer_name_or_path $model_name --test_batch_size $batchsize --pct_train_rationales $pct_train_rationales --nli_trained $nli_trained 
    ## Ours (soft) 
    # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --soft True
    # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --soft True --load_model True --evaluate_robustness True

done

