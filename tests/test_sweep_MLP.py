"""
In this test we will try to use the sweep feature of Weights & Biases to find the best
hyperparameters for our NN. 

We will use the same NN as in the simpleNNexample.py script, but we will try to find the best
number of hidden neurons and the best activation function for the hidden layer, as well as the
best optimizer, batch size and learning rate.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

import path_helper

from src.io import load_and_norm_data, process_data
from src.models import MLP, train_MLP

import gc


def sweep_MLP():
    # Initialize Weights & Biases run
    run = wandb.init(config=sweep_config)

    # Process data
    train_dataset, test_dataset = process_data(
        N, NUMBER_OF_PULSES, pulse_dataset, wandb.config.training_size, wandb.config.batch_size)

    # Build the model with the config
    model = MLP(wandb.config.input_shape, wandb.config.output_shape, wandb.config.n_hidden_layers,
            wandb.config.n_neurons_per_layer, wandb.config.activation, wandb.config.dropout)

    # Print the model summary
    model.summary()

    # Set the optimizer with the config with its learning rate
    optimizer = keras.optimizers.get(wandb.config.optimizer)
    optimizer.learning_rate = wandb.config.learning_rate

    # Train the model with the config
    train_MLP(train_dataset, test_dataset, model,
            epochs=wandb.config.epochs,
            optimizer=optimizer,
            loss_fn=keras.losses.get(wandb.config.loss),
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
            'method': 'bayes',
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
                    'values': ['adam', 'sgd']
                },
                'learning_rate': {
                    'values': [0.1, 0.01, 0.001, 0.0001]
                },
                'n_hidden_layers': {
                    'values': [1, 2, 3, 4, 5]
                },
                'n_neurons_per_layer': {
                    'values': [512, 1024, 2048, 3072, 4096]
                },
                'activation': {
                    'values': ['sigmoid', 'tanh', 'relu', 'elu', 'silu', 'swish', 'gelu']
                },
                'dropout': {
                    'values': [None, 0.1, 0.2, 0.3, 0.4, 0.5]
                },
                'patience': {
                    'values': [10]
                },
                'training_size': {
                    'values': [0.75]
                },
                'database': {
                    'values': [f'{NUMBER_OF_PULSES}_randomPulses_N{N}']
                },
                'arquitecture': {
                    'values': ['MLP']
                },
                'input_shape': {
                    'values': [(N, N)]
                },
                'output_shape': {
                    'values': [(int(2 * N))]
                },
                'loss': {
                    'values': ['mse', 'mae', 'log_cosh']
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
    sweep_id = wandb.sweep(sweep_config, project="Corrected data MLP sweep")

    # Memory cleanup code. This is necessary because between runs, TensorFlow does not release GPU and the memory is not freed.
    def sweep_MLP_with_cleanup():
        try:
            # Run the sweep
            sweep_MLP()
        finally:
            # Cleanup code
            tf.keras.backend.clear_session()  # Clear the TensorFlow session
            gc.collect()  # Force garbage collector to release unreferenced memory


    wandb.agent(sweep_id, function=sweep_MLP_with_cleanup, count=50)