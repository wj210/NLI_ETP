
export OMP_NUM_THREADS=8 
model_name='roberta-base'
batchsize=16
export CUDA_VISIBLE_DEVICES=0,1,2,3
ds=fever
nli_trained=True
for pct_train_rationales in 0.1
do  
    for seed in 42 43 44
    do
        # blackbox model , set plaus_weight to 0
        # python main.py --gpu_no 1 --tasks $ds --plaus_weight 0.0 --seed $seed  
        # python main.py --gpu_no 1 --tasks $ds --plaus_weight 0.0 --seed $seed --load_model True

        # # blackbox model , set supervised to True, plaus to 1.0
        # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --supervised True
        # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --supervised True --load_model True --evaluate_robustness True

        ## Ours (hard) 
        python main.py --gpu_no 0 1 2 3 --tasks $ds --plaus_weight 1.0 --seed $seed --min_one True --model_name_or_path $model_name --tokenizer_name_or_path $model_name --train_batch_size $batchsize --eval_batch_size $batchsize --pct_train_rationales $pct_train_rationales --nli_trained $nli_trained
        python main.py --gpu_no 0 --tasks $ds --plaus_weight 1.0 --seed $seed --min_one True --model_name_or_path $model_name --tokenizer_name_or_path $model_name --test_batch_size $batchsize --pct_train_rationales $pct_train_rationales --nli_trained $nli_trained --align True --load_model True
        ## Ours (soft) 
        # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --soft True
        # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --soft True --load_model True --evaluate_robustness True

    done
done

# for seed in 42 43 44
# do
#     # blackbox model , set plaus_weight to 0
#     # python main.py --gpu_no 1 --tasks $ds --plaus_weight 0.0 --seed $seed  
#     # python main.py --gpu_no 1 --tasks $ds --plaus_weight 0.0 --seed $seed --load_model True

#     # # blackbox model , set supervised to True, plaus to 1.0
#     # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --supervised True
#     # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --supervised True --load_model True --evaluate_robustness True

#     ## Ours (hard) 
#     python main.py --gpu_no 0 --tasks $ds --plaus_weight 1.0 --seed $seed --min_one True --model_name_or_path $model_name --tokenizer_name_or_path $model_name --train_batch_size $batchsize --eval_batch_size $batchsize --pct_train_rationales 0.1
#     python main.py --gpu_no 0 --tasks $ds --plaus_weight 1.0 --seed $seed --min_one True --load_model True  --model_name_or_path $model_name --tokenizer_name_or_path $model_name --test_batch_size $batchsize --pct_train_rationales 0.1  
#     ## Ours (soft) 
#     # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --soft True
#     # python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --soft True --load_model True --evaluate_robustness True

# done