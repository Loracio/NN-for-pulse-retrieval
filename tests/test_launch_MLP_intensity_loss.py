"""
In this example, we will use Weights & Biases to log the training process of a MLP using a custom loss function (intensity loss).

The Neural Network will try to predict the Electric field of a pulse in the time domain from its SHG-FROG trace.
The input data is the SHG-FROG trace of the pulse (NxN vector), and the target data is the
Electric field of the pulse in the time domain (2N-dimensional vector containing
the real and imaginary parts of the pulse).

The loss function is defined in src/models/custom_loss.py and it is the mean squared error between the
predicted and the target intensities in time and frequency domains.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

import path_helper

from src.io import process_data_tfrecord
from src.models import MLP, bottleneck_MLP, train_MLP_intensity_loss, intensity_loss

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
        'epochs': 100,
        'batch_size': 256,
        'log_step': 50,
        'val_log_step': 50,
        'optimizer': 'adam',
        'learning_rate': 0.1,
        'loss': 'intensity_loss',
        'train_metrics': 'MeanSquaredError',
        'val_metrics': 'MeanSquaredError',
        'n_hidden_layers': 2,
        'n_neurons_per_layer': 512,
        'reduction_factor': 2,
        'activation': 'relu',
        'dropout': 0.05,
        'patience': 10,
        'training_size': 0.85,
        'arquitecture': 'bottleneck_MLP',
        'input_shape': (N, N),
        'output_shape': (int(2 * N)),
    }

    # Load and process the pulse database
    train_dataset, test_dataset = process_data_tfrecord(
        N, NUMBER_OF_PULSES, FILE_PATH, config['training_size'], config['batch_size'])

    # Initialize Weights & Biases with the config parameters
    run = wandb.init(project="MSE field vs intensity", config=config,
                     name='Bottleneck MLP MSE FIELD')

    # Build the model with the config
    if config['arquitecture'] == 'MLP':
        model = MLP(config['input_shape'], config['output_shape'], config['n_hidden_layers'],
                config['n_neurons_per_layer'], config['activation'], config['dropout'])

    if config['arquitecture'] == 'bottleneck_MLP':
        model = bottleneck_MLP(config['input_shape'], config['output_shape'], config['n_hidden_layers'],
                           config['n_neurons_per_layer'], config['reduction_factor'], config['activation'], config['dropout'])


    # Print the model summary
    model.summary()

    # Set the optimizer with the config with its learning rate
    optimizer = keras.optimizers.get(config['optimizer'])
    optimizer.learning_rate = config['learning_rate']

    intensity_loss = intensity_loss(N, 1/N)

    # Train the model with the config
    train_MLP_intensity_loss(train_dataset, test_dataset, model,
                          epochs=config['epochs'],
                          optimizer=optimizer,
                          intensity_loss_fn=intensity_loss,
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

    # Save the model
    # model.save(f"./trained_models/FCNN/{config['arquitecture']}_test1.h5")

    # # open model
    # model = keras.models.load_model(
    #     f"./trained_models/FCNN/{config['arquitecture']}_test1.h5")

    import matplotlib.pyplot as plt

    # Show the results
    result_test = resultsGUI(model, test_dataset, int(0.15 * NUMBER_OF_PULSES), N, 1/N, norm_predictions=True)
    result_test.plot()

    plt.show()