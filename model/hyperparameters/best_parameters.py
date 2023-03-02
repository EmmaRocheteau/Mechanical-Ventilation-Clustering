task_weighting = {
    'duration_weight': 0.015,
    'binary_weight': 0.02,
    'categorical_weight': 0.01,
    'reconstruction_weight': 0.02
}
encoder={
    'lstm':
        {'batch_size': 128,
         'lr': 0.005,
         'l2': 0.00052,
         'dropout': 0,
         'num_layers_lstm': 2,
         'h_dim_lstm': 128,
         'load_encoder': 'default/best_lstm',
         'encoder_epochs': 28  # this should be the same as the epoch number of the saved checkpoint to be loaded
         },
    'tpc':
        {'batch_size': 128,
         'lr': 0.0045,
         'l2': 0.00053,
         'dropout': 0.05,
         'num_layers_tpc': 6,
         'kernel_size': 3,
         'no_temp_kernels': 6,
         'point_size': 14,
         'temp_dropout_rate': 0.02,
         'last_linear_size': 16,  # the output is now the same size as the output of the lstm
         'encoding_dim_tpc': 128,
         'load_encoder': 'default/best_tpc',
         'encoder_epochs': 47  # this should be the same as the epoch number of the saved checkpoint to be loaded
         }}
decoder={
    'h_dim_decoder': 50  # this is for tasks which use 2-layer decoders
}