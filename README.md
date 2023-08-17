# NLI_ETP

1. Download the Eraser data
2. Preprocess data to train NLI predictor
3. Train NLI predictor
4. Preprocess base dataset to train explainer and predictor
5. Train explainer and predictor

Step 1 - run scripts/dl_eraser.sh

Step 2-4 - run scripts/preprocess_data.sh

step 5 - run scripts/main.sh (includes Full-C , Supervised , Hard and Soft approach for 3 datasets)

ROBUSTNESS TEST
To get data to carry out robustness, ie get the adversarial sentences, change the root and data directory path in rationale-robustness/rr/config.py to your specified path.

rationale-robustness is from https://github.com/princeton-nlp/rationale-robustness

ERASER dataset is from https://github.com/jayded/eraserbenchmark

To get averaged results across all seed runs, run get_seed_results.py
