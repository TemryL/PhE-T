# Model:
bert_config = {
    'p_size': 25,
    'v_size': 665,
    'context_size': 24,
    'n_layers': 12,
    'n_heads': 12,
    'h_dim': 768,
    'ln_eps': 1e-12,
    'dropout': 0.1
}

# Data:
batch_size = 256
train_val_split = 0.95
mlm_probability = 0.15
n_bins = 100
train_data = 'data/train.csv'
test_data = 'data/test.csv'
num_features = [
    'BMI',
    'HDL cholesterol',
    'LDL cholesterol',
    'Total cholesterol',
    'Triglycerides',
    'Diastolic blood pressure'
]
cat_features = [
    'Age',
    'Sex',
    'Ever smoked',
    'Snoring',
    'Insomnia',
    'Daytime napping',
    'Chronotype',
    'Sleep duration',
]
diseases = [
    'Asthma',
    'Cataract',
    'Diabetes',
    'GERD',
    'Hay-fever & Eczema',
    'Major depression',
    'Myocardial infarction',
    'Osteoarthritis',
    'Pneumonia',
    'Stroke'
]

# Optimization:
learning_rate = 1e-4
adamw_epsilon = 1e-6
adamw_betas = (0.9, 0.98)
warmup_steps = 10000
weight_decay = 0.01