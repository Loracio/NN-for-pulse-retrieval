"""
In this test we will try to use the sweep feature of Weights & Biases to find the best
hyperparameters for our NN. 

We will be using the custom loss function defined in src/models/custom_loss.py and the
train_MLP_custom_loss function defined in src/models/train_model.py to train the model.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

import path_helper

from src.io import load_and_norm_data, process_data
from src.models import MLP, bottleneck_MLP, train_MLP_custom_loss, trace_loss

import gc


def sweep_MLP():
    # Initialize Weights & Biases run
    run = wandb.init(config=sweep_config)

    # Process data
    train_dataset, test_dataset = process_data(
        N, NUMBER_OF_PULSES, pulse_dataset, wandb.config.training_size, wandb.config.batch_size)

    # Build the model with the config, depending on the arquitecture
    if wandb.config.arquitecture == 'bottleneck_MLP':
        #! Reduction factor set to 2 always
        model = bottleneck_MLP(wandb.config.input_shape, wandb.config.output_shape, wandb.config.n_hidden_layers,
                               wandb.config.n_neurons_per_layer, 2, wandb.config.activation, wandb.config.dropout)
    if wandb.config.arquitecture == 'MLP':
        model = MLP(wandb.config.input_shape, wandb.config.output_shape, wandb.config.n_hidden_layers,
                    wandb.config.n_neurons_per_layer, wandb.config.activation, wandb.config.dropout)

    # Print the model summary
    model.summary()

    # Set the optimizer with the config with its learning rate
    optimizer = keras.optimizers.get(wandb.config.optimizer)
    optimizer.learning_rate = wandb.config.learning_rate

    # Trace loss
    custom_loss = trace_loss(N, 1/N)

    # Train the model with the config
    train_MLP_custom_loss(train_dataset, test_dataset, model,
              epochs=wandb.config.epochs,
              optimizer=optimizer,
              custom_loss_fn=custom_loss,
              train_acc_metric=keras.metrics.get(wandb.config.train_metrics),
              test_acc_metric=keras.metrics.get(wandb.config.val_metrics),
              log_step=wandb.config.log_step,
              val_log_step=wandb.config.val_log_step,
              patience=wandb.config.patience
              )


if __name__ == "__main__":
    # Define pulse database parameters
    N = 64
    NUMBER_OF_PULSES = 2500
    FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"
    # Handle error if path does not exist
    try:
        with open(FILE_PATH) as f:
            pass
    except FileNotFoundError:
        print("File not found. Please generate the pulse database first.")
        exit()

    # Load dataset
    pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)

    # Define config parameters for wandb
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'epochs': {
                'values': [50]
            },
            'batch_size': {
                'values': [16, 32, 64, 128, 256]
            },
            'log_step': {
                'values': [50]
            },
            'val_log_step': {
                'values': [50]
            },
            'optimizer': {
                'values': ['adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']
            },
            'learning_rate': {
                'values': [0.1, 0.01, 0.001, 0.0001]
            },
            'n_hidden_layers': {
                'values': [1, 2, 3, 4, 5]
            },
            'n_neurons_per_layer': {
                'values': [256, 512, 1024, 2048, 3072, 4096]
            },
            'activation': {
                'values': ['sigmoid', 'tanh', 'relu', 'elu', 'silu', 'swish', 'gelu']
            },
            'dropout': {
                'values': [None, 0.1, 0.2, 0.3]
            },
            'patience': {
                'values': [5]
            },
            'training_size': {
                'values': [0.8]
            },
            'database': {
                'values': [f'{NUMBER_OF_PULSES}_randomPulses_N{N}']
            },
            'arquitecture': {
                'values': ['MLP', 'bottleneck_MLP']
            },
            'input_shape': {
                'values': [(N, N)]
            },
            'output_shape': {
                'values': [(int(2 * N))]
            },
            'loss': {
                'values': ['trace_loss']
            },
            'train_metrics': {
                'values': ['MeanSquaredError']
            },
            'val_metrics': {
                'values': ['MeanSquaredError']
            },

        }
    }

    # Initialize Weights & Biases sweep with the config parameters
    sweep_id = wandb.sweep(sweep_config, project="Custom loss MLP sweep")

    # Memory cleanup code. This is necessary because between runs.
    def sweep_MLP_with_cleanup():
        try:
            # Run the sweep
            sweep_MLP()
        finally:
            # Cleanup code
            tf.keras.backend.clear_session()  # Clear the TensorFlow session
            gc.collect()  # Force garbage collector to release unreferenced memory

    wandb.agent(sweep_id, function=sweep_MLP_with_cleanup, count=100)
