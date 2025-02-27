split_parameters = {
    'method' : 'depid',   #'depid', 'random_split', 'temporal_split' or 'skf'
    'split' : 0.8,    # Fraction of data used for training set. Selected randomly or in temporal order.
    'n_splits' : 5,   # Number of splits for skf split
    'test_depid' : 'ml19_293a'  # Depid used as testing set
}

normalization = {
    'acoustic_min' : -90,
    'acoustic_max' : 40
}

model_params = {
    'input_size' : 513,
    'hidden_size' : 1024
}

hyperparameters = {
    'learning_rate' : 0.0001,
    'batch_size' : 64,
    'weight_decay' : 0.000,
    'num_epochs' : 30
}
