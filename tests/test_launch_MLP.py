"""
In this example, we will use Weights & Biases to log the training process of a MLP.

The Neural Network will try to predict the Electric field of a pulse in the time domain from its SHG-FROG trace.
The input data is the SHG-FROG trace of the pulse (NxN vector), and the target data is the
Electric field of the pulse in the time domain (2N-dimensional vector containing
the real and imaginary parts of the pulse).
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

import path_helper

from src.io import load_and_norm_data, process_data
from src.models import MLP, train_MLP

if __name__ == "__main__":
    # Define pulse database parameters
    N = 64
    NUMBER_OF_PULSES = 1000
    FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"
    # Handle error if path does not exist
    try:
        with open(FILE_PATH) as f:
            pass
    except FileNotFoundError:
        print("File not found. Please generate the pulse database first.")
        exit()

    # Define config parameters for wandb
    config = {
        'epochs': 25,
        'batch_size': 256,
        'log_step': 50,
        'val_log_step': 50,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'mse',
        'train_metrics': 'MeanSquaredError',
        'val_metrics': 'MeanSquaredError',
        'n_hidden_layers': 2,
        'n_neurons_per_layer': 3072,
        'activation': 'relu',
        'dropout': None,
        'patience': 10,
        'training_size': 0.75,
        'database': f'{NUMBER_OF_PULSES}_randomPulses_N{N}',
        'arquitecture': 'MLP',
        'input_shape': (N, N),
        'output_shape': (int(2 * N)),
    }

    # Load and process the pulse database
    pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)
    train_dataset, test_dataset = process_data(N, NUMBER_OF_PULSES, pulse_dataset,
                                            config['training_size'], config['batch_size'])

    # Initialize Weights & Biases with the config parameters
    run = wandb.init(project="MLP_example", config=config,
                    name='MLP test run #3')

    # Build the model with the config
    model = MLP(config['input_shape'], config['output_shape'], config['n_hidden_layers'],
                config['n_neurons_per_layer'], config['activation'], config['dropout'])

    # Print the model summary
    model.summary()

    # Set the optimizer with the config with its learning rate
    optimizer = keras.optimizers.get(config['optimizer'])
    optimizer.learning_rate = config['learning_rate']

    # Train the model with the config
    train_MLP(train_dataset, test_dataset, model,
                epochs=config['epochs'],
                optimizer=optimizer,
                loss_fn=keras.losses.get(config['loss']),
                train_acc_metric=keras.metrics.get(config['train_metrics']),
                test_acc_metric=keras.metrics.get(config['val_metrics']),
                log_step=config['log_step'],
                val_log_step=config['val_log_step'],
                patience=config['patience']
                )

    # Finish the run
    run.finish()

    # Save the model
    # model.save(f"./trained_models/FCNN/{config['arquitecture']}_test2.h5")