# NLI_ETP

Download the requirements using pip install -r requirements.txt

Replace the model_checkpoint.py from pytorch_lightning package in your conda environment with the model_checkpoint. Just so that the overwrite_existing can be used in the ModelCheckpoint in main.py. Just to prevent saving multiple same models which can take up alot of space. If this is not a problem, can ignore.

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
