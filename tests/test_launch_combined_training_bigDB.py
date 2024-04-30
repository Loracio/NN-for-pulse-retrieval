"""
In this example, we will use Weights & Biases to log the training process of a CNN using a 
custom training process in which the data are first trained on the MSE of the field and then
on the trace loss.

The Neural Network will try to predict the Electric field of a pulse in the time domain from its SHG-FROG trace.
The input data is the SHG-FROG trace of the pulse (NxN vector), and the target data is the
Electric field of the pulse in the time domain (2N-dimensional vector containing
the real and imaginary parts of the pulse).

We will use a very large database of pulses to train the model, so when doing the training
step, several files will be opened and then closed to avoid memory issues.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

import path_helper

from src.io import process_data_tfrecord
from src.models import CNN, MultiResNet, DenseNet, trace_loss, trace_MSE, train_combined_loss_training_BD
from src.visualization import resultsGUI

if __name__ == "__main__":
    # Define pulse database parameters
    N = 128
    NUMBER_OF_PULSES = 20000
    BATCH_SIZE = 1024

    NUMBER_OF_BATCHES = np.ceil(NUMBER_OF_PULSES / BATCH_SIZE).astype(int)

    trainFILENAMES = []

    VALIDATION_PULSES = 1000
    VALIDATION_BATCHES = np.ceil(VALIDATION_PULSES / BATCH_SIZE).astype(int)

    validationFILENAMES = []

    for batch in range(NUMBER_OF_BATCHES):
        trainFILENAMES.append(f"./data/generated/N{N}/{NUMBER_OF_PULSES}_batchedPulses/train/{NUMBER_OF_PULSES}_randomNormalizedPulses_N{N}_batch{batch}.tfrecords")

    for batch in range(VALIDATION_BATCHES):
        validationFILENAMES.append(f"./data/generated/N{N}/{NUMBER_OF_PULSES}_batchedPulses/validation/{NUMBER_OF_PULSES}_randomNormalizedPulses_N{N}_batch{batch}.tfrecords")

    # Define config parameters for wandb
    config = {
        'start_with': 0,  # 0 : Start with the field loss, 1 : Start with the trace loss
        'trace_epochs': 250,
        'field_epochs': 50,
        'reps': 1,  # Number of repetitions of the combined training
        'batch_size': BATCH_SIZE,
        'log_step': 200,
        'val_log_step': 200,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'trace_loss',
        'n_conv_layers': 2,  # Number of convolutional layers
        'n_filters_per_layer': 32,  # Number of filters per layer
        # Reduction factor for the number of filters in each layer
        'reduce_filter_factor': 0.25,
        'kernel_size': (3, 3),  # Kernel size
        'pool': True,  # Use pooling layers
        'pool_size': (2, 2),  # Pool size
        'conv_activation': 'relu',  # Activation function for the convolutional layers
        'n_dense_layers': 2,  # Number of dense layers
        'n_neurons_per_layer': 512,  # Number of neurons per dense layer
        # Reduction factor for the number of neurons in each layer in the dense layers
        'reduce_dense_factor': 2,
        'dense_activation': 'relu',  # Activation function for the dense layers
        'dropout': 0.05,  # Dropout rate, if None, no dropout is used
        'patience': 100,  # Patience for the early stopping
        'validation_pulses': VALIDATION_PULSES,
        'database': f'{NUMBER_OF_PULSES}_randomPulses_N{N}',
        'arquitecture': 'CNN',  # 'MultiResNet', 'DenseNet', 'CNN
        # The number of channels is the last element of the input shape
        'input_shape': (N, N, 1),
        'output_shape': (int(2 * N)),
    }

    # Initialize Weights & Biases with the config parameters
    run = wandb.init(project="MSE field vs intensity", config=config,
                     name='Test BIG DATABASE',)

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

    trace_loss_fn = trace_loss(N, 1/N)
    field_loss_fn = keras.losses.MeanSquaredError()

    train_trace_metric = trace_MSE(N, 1/N)
    train_field_metric = keras.metrics.MeanSquaredError()

    test_trace_metric = trace_MSE(N, 1/N)
    test_field_metric = keras.metrics.MeanSquaredError()

    # Train the model with the config
    train_combined_loss_training_BD(trainFILENAMES,
                                 validationFILENAMES,
                                 N,
                                 BATCH_SIZE,
                                 model,
                                 optimizer,
                                 trace_loss_fn,
                                 field_loss_fn,
                                 train_trace_metric,
                                 train_field_metric,
                                 test_trace_metric,
                                 test_field_metric,
                                 config['field_epochs'],
                                 config['trace_epochs'],
                                 config['start_with'],
                                 config['reps'],
                                 config['log_step'],
                                 config['val_log_step'],
                                 config['patience'])

    # Finish the run
    run.finish()

    # Save the model using tensorflow save method
    model.save(
        f"./trained_models/CNN/{config['arquitecture']}_test_N{N}_normTraces_combinedTraining_BIGDB.tf")
