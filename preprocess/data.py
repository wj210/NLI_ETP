dataset_info = {
    'movies': {
        'train': ['train', 1599],
        'dev': ['val', 200],
        'test': ['test', 200],
        'num_classes': 2,
        'classes': ['NEG', 'POS'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 1024,
            'google/bigbird-roberta-large': 1024,
            'roberta-base': 512,
            'roberta-large': 512,
        },
        'num_special_tokens': 2,
    },
    'multirc': {
        'train': ['train', 24029],
        'dev': ['val', 3214],
        'test': ['test', 4848],
        'num_classes': 2,
        'classes': ['False', 'True'],
        'top_p': 0.25,
        'top_k':0,
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 748,
            'roberta-base': 512,
            'roberta-large': 512,
        },
        'num_special_tokens': 3,
    },
    'boolq': {
        'train': ['train', 6363],
        'dev': ['val', 1491],
        'test': ['test', 2807],
        'num_classes': 2,
        'classes': ['False', 'True'],
        'top_p': 0.2,
        'top_k':1,
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 2048,
            'google/bigbird-roberta-large': 2048,
            'roberta-base': 512,
            'roberta-large': 512,
        },
        'num_special_tokens': 3,
    },
    'fever': {
        'train': ['train', 97957],
        'dev': ['val', 6122],
        'test': ['test', 6111],
        'num_classes': 2,
        'classes': ['REFUTES', 'SUPPORTS'],
        'top_p': 0.4,
        'top_k':1,
        'max_sentences': 10,
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 1024,
            'google/bigbird-roberta-large': 1024,
            'roberta-base': 512,
            'roberta-large': 512
        },
        'num_special_tokens': 3,
    },
    'evidence_inference': {
        'train': ['train', 7958],
        'dev': ['val', 972],
        'test': ['test', 959],
        'num_classes': 3,
        'classes': ['significantly decreased', 'significantly increased','no significant difference'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 2048,
            'google/bigbird-roberta-large': 2048,
            'roberta-base': 512,
        },
        'num_special_tokens': 5,
    },
}

## RTXA6000 46GB
dataset_batch_size = {'multirc':8, 'boolq':8,'evidence_inference':4,'fever':8,'movies':8}

data_keys = ['input_ids', 'attention_mask', 'rationale_token', 'rationale_sen', 'sen_marker', 'label','sen_has_rationale','tok_has_rationale']