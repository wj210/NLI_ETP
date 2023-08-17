
export OMP_NUM_THREADS=8 
for ds in boolq fever multirc
do  
    for seed in 42 43 44
    do
        # blackbox model , set plaus_weight to 0
        python main.py --gpu_no 1 --tasks $ds --plaus_weight 0.0 --seed $seed  
        python main.py --gpu_no 1 --tasks $ds --plaus_weight 0.0 --seed $seed --load_model True

        # blackbox model , set supervised to True, plaus to 1.0
        python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --supervised True
        python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --supervised True --load_model True --evaluate_robustness True

        ## Ours (hard) 
        python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --min_one True
        python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --min_one True --load_model True --evaluate_robustness True

        ## Ours (soft) 
        python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --soft True
        python main.py --gpu_no 1 --tasks $ds --plaus_weight 1.0 --seed $seed --soft True --load_model True --evaluate_robustness True

    done

done