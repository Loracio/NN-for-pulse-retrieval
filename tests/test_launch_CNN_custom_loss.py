"""
In this example, we will use Weights & Biases to log the training process of a CNN using a custom loss function.

The Neural Network will try to predict the Electric field of a pulse in the time domain from its SHG-FROG trace.
The input data is the SHG-FROG trace of the pulse (NxN vector), and the target data is the
Electric field of the pulse in the time domain (2N-dimensional vector containing
the real and imaginary parts of the pulse).

The loss function is defined in src/models/custom_loss.py and it is the mean squared error between the
predicted and the target SHG-FROG trace.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

import path_helper

from src.io import load_and_norm_data, process_data
from src.models import CNN, train_CNN_custom_loss, trace_loss, MultiResNet, DenseNet
from src.visualization import resultsGUI

if __name__ == "__main__":
    # Define pulse database parameters
    N = 128
    NUMBER_OF_PULSES = 5000
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
        'epochs': 250,
        'batch_size': 32,
        'log_step': 200,
        'val_log_step': 200,
        'optimizer': 'adam',
        'learning_rate': 0.005,
        'loss': 'trace_loss',
        'train_metrics': 'MeanSquaredError',
        'val_metrics': 'MeanSquaredError',
        'n_conv_layers': 2, # Number of convolutional layers
        'n_filters_per_layer': 32, # Number of filters per layer
        'reduce_filter_factor': 0.25, # Reduction factor for the number of filters in each layer
        'kernel_size': (3, 3), # Kernel size
        'pool': True, # Use pooling layers
        'pool_size': (2, 2), # Pool size
        'conv_activation': 'sigmoid', # Activation function for the convolutional layers
        'n_dense_layers': 4, # Number of dense layers
        'n_neurons_per_layer': 2048, # Number of neurons per dense layer
        'reduce_dense_factor': 2, # Reduction factor for the number of neurons in each layer in the dense layers
        'dense_activation': 'sigmoid', # Activation function for the dense layers
        'dropout': 0.05, # Dropout rate, if None, no dropout is used
        'patience': 15, # Patience for the early stopping
        'training_size': 0.8,
        'database': f'{NUMBER_OF_PULSES}_randomPulses_N{N}',
        'arquitecture': 'CNN', # 'MultiResNet', 'DenseNet', 'CNN
        'input_shape': (N, N, 1), # The number of channels is the last element of the input shape
        'output_shape': (int(2 * N)),
    }

    # Load and process the pulse database
    pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)
    train_dataset, test_dataset = process_data(N, NUMBER_OF_PULSES, pulse_dataset,
                                               config['training_size'], config['batch_size'])

    # Initialize Weights & Biases with the config parameters
    run = wandb.init(project="N=128", config=config,
                     name='CNN test with tf.function decirator')

    # Build the model with the config
    if config['arquitecture'] == 'MultiResNet':
        model = MultiResNet()

    if config['arquitecture'] == 'DenseNet':
        model = DenseNet([6, 12, 24, 16])

    if config['arquitecture'] == 'CNN':
        model = CNN(config['input_shape'], config['output_shape'],
                n_conv_layers=config['n_conv_layers'],
                n_filters_per_layer=config['n_filters_per_layer'],
                reduce_filter_factor=config['reduce_filter_factor'],
                kernel_size=config['kernel_size'],
                pool=config['pool'],
                pool_size=config['pool_size'],
                conv_activation=config['conv_activation'],
                n_dense_layers=config['n_dense_layers'],
                n_neurons_per_layer=config['n_neurons_per_layer'],
                reduce_dense_factor=config['reduce_dense_factor'],
                dense_activation=config['dense_activation'],
                dropout=config['dropout'])


    # Print the model summary
    model.summary()

    # Set the optimizer with the config with its learning rate
    optimizer = keras.optimizers.get(config['optimizer'])
    optimizer.learning_rate = config['learning_rate']

    custom_loss = trace_loss(N, 1/N)

    # Train the model with the config
    train_CNN_custom_loss(train_dataset, test_dataset, model,
                          epochs=config['epochs'],
                          optimizer=optimizer,
                          custom_loss_fn=custom_loss,
                          train_acc_metric=keras.metrics.get(
                              config['train_metrics']),
                          test_acc_metric=keras.metrics.get(
                              config['val_metrics']),
                          log_step=config['log_step'],
                          val_log_step=config['val_log_step'],
                          patience=config['patience']
                          )

    # Finish the run
    run.finish()

    # Save the model using tensorflow save method
    model.save(f"./trained_models/CNN/{config['arquitecture']}_test_N{N}.tf")

    # import matplotlib.pyplot as plt

    # # Show the results
    # result_test = resultsGUI(model, test_dataset, int(len(test_dataset)), N, 1/N)
    # result_test.plot()

    # plt.show()