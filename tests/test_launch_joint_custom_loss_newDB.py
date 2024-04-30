"""
In this example, we will use Weights & Biases to log the training process of a CNN using a custom loss function
that takes into account the MSE in the trace and the MSE in the electric field.

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

from src.io import process_data_tfrecord
from src.models import CNN, MultiResNet, DenseNet, trace_loss, trace_MSE, intensity_loss, intensity_MSE, train_joint_loss_intensity
from src.visualization import resultsGUI

if __name__ == "__main__":
    # Define pulse database parameters
    N = 128
    NUMBER_OF_PULSES = 1000
    FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomNormalizedPulses_N{N}.tfrecords"
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
        'batch_size': 256,
        'log_step': 200,
        'val_log_step': 200,
        'optimizer': 'adam',
        'learning_rate': 1e-3,
        'loss': 'trace_loss',
        'weight_trace_loss': 1,  # Weight for the trace loss
        'weight_intensity_loss': 1,  # Weight for the mse loss
        'n_conv_layers': 2,  # Number of convolutional layers
        'n_filters_per_layer': 32,  # Number of filters per layer
        # Reduction factor for the number of filters in each layer
        'reduce_filter_factor': 0.25,
        'kernel_size': (3, 3),  # Kernel size
        'pool': True,  # Use pooling layers
        'pool_size': (2, 2),  # Pool size
        'conv_activation': 'relu',  # Activation function for the convolutional layers
        'n_dense_layers': 3,  # Number of dense layers
        'n_neurons_per_layer': 512,  # Number of neurons per dense layer
        # Reduction factor for the number of neurons in each layer in the dense layers
        'reduce_dense_factor': 2,
        'dense_activation': 'relu',  # Activation function for the dense layers
        'dropout': 0.05,  # Dropout rate, if None, no dropout is used
        'patience': 15,  # Patience for the early stopping
        'training_size': 0.9,
        'database': f'{NUMBER_OF_PULSES}_randomPulses_N{N}',
        'arquitecture': 'CNN',  # 'MultiResNet', 'DenseNet', 'CNN
        # The number of channels is the last element of the input shape
        'input_shape': (N, N, 1),
        'output_shape': (int(2 * N)),
    }

    # Load and process the pulse database
    train_dataset, test_dataset = process_data_tfrecord(
        N, NUMBER_OF_PULSES, FILE_PATH, config['training_size'], config['batch_size'], add_noise=False, noise_level=0.01, mask=True, mask_tolerance=1e-3)

    # Initialize Weights & Biases with the config parameters
    run = wandb.init(project="MSE field vs intensity", config=config,
                     name='Joint loss intensity',)

    # Build the model with the config
    if config['arquitecture'] == 'MultiResNet':
        model = MultiResNet()

    if config['arquitecture'] == 'DenseNet':
        model = DenseNet([6, 12, 24, 16], input_shape=config['input_shape'],
                         output_shape=config['output_shape'])

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
    intensity_loss_fn = intensity_loss(N, 1/N)

    train_trace_metric = trace_MSE(N, 1/N)
    train_field_metric = keras.metrics.MeanSquaredError()
    train_intensity_metric = intensity_MSE(N, 1/N)

    test_trace_metric = trace_MSE(N, 1/N)
    test_field_metric = keras.metrics.MeanSquaredError()
    test_intensity_metric = intensity_MSE(N, 1/N)

    # Train the model with the config
    train_joint_loss_intensity(train_dataset,
                               test_dataset,
                               model,
                               optimizer,
                               config['weight_trace_loss'],
                               trace_loss_fn,
                               config['weight_intensity_loss'],
                               intensity_loss_fn,
                               train_trace_metric,
                               train_field_metric,
                               test_trace_metric,
                               test_field_metric,
                               train_intensity_metric,
                               test_intensity_metric,
                               config['epochs'],
                               config['log_step'],
                               config['val_log_step'],
                               config['patience']
                               )

    # Finish the run
    run.finish()

    # # Save the model using tensorflow save method
    # model.save(
    #     f"./trained_models/CNN/{config['arquitecture']}_test_N{N}_normTraces_total_TESTNOISE.tf")

    # model = keras.models.load_model(
    #     f"./trained_models/CNN/{config['arquitecture']}_test_N{N}_normTraces_total_TESTNOISE.tf")

    # import matplotlib.pyplot as plt
    # import scienceplots

    # plt.rcParams.update({'font.size': 16})
    # plt.style.use('science')
    # plt.rcParams['figure.figsize'] = [6, 4]
    # plt.rcParams['legend.frameon'] = True
    # plt.rcParams['legend.fancybox'] = True
    # plt.rcParams['axes.grid'] = True
    # plt.rcParams['grid.alpha'] = 0.5
    # plt.rcParams['grid.linestyle'] = '--'
    # plt.rcParams['grid.linewidth'] = 0.5
    # plt.rcParams['axes.axisbelow'] = True

    # # Show the results
    # result_test = resultsGUI(model, test_dataset, int(0.1 * NUMBER_OF_PULSES), N, 1/N, norm_predictions=True)
    # result_test.plot()

    # plt.show()