"""
In this example, we will use Weights & Biases to log the training process of a MLP using a custom loss function.

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
from src.models import MLP, bottleneck_MLP, train_MLP_custom_loss, trace_loss

from src.utils import compute_trace

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

    # Define config parameters for wandb
    config = {
        'epochs': 25,
        'batch_size': 256,
        'log_step': 50,
        'val_log_step': 50,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'trace_loss',
        'train_metrics': 'MeanSquaredError',
        'val_metrics': 'MeanSquaredError',
        'n_hidden_layers': 4,
        'n_neurons_per_layer': 2048,
        'reduction_factor': 2,
        'activation': 'elu',
        'dropout': None,
        'patience': 10,
        'training_size': 0.75,
        'database': f'{NUMBER_OF_PULSES}_randomPulses_N{N}',
        'arquitecture': 'bottleneck_MLP',
        'input_shape': (N, N),
        'output_shape': (int(2 * N)),
    }

    # Load and process the pulse database
    pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)
    train_dataset, test_dataset = process_data(N, NUMBER_OF_PULSES, pulse_dataset,
                                               config['training_size'], config['batch_size'])

    # Initialize Weights & Biases with the config parameters
    run = wandb.init(project="Custom loss tests", config=config,
                     name='MLP custom trace tanh activation')

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

    custom_loss = trace_loss(N, 1/N)

    # Train the model with the config
    train_MLP_custom_loss(train_dataset, test_dataset, model,
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

    # Save the model
    # model.save(f"./trained_models/FCNN/{config['arquitecture']}_test1.h5")

    # # open model
    # model = keras.models.load_model(
    #     f"./trained_models/FCNN/{config['arquitecture']}_test1.h5")

    import matplotlib.pyplot as plt

    # Plot the predicted SHG-FROG trace of the first pulse in the test dataset
    for x, y in iter(test_dataset):
        y_pred = model.predict(x)
        y_pred = y_pred[0]
        y_pred_complex = y_pred[:N] + 1j * y_pred[N:]
        t = np.array([-np.floor(0.5 * N) / N + i / N for i in range(N)])
        pred_trace = compute_trace(y_pred_complex, t, 1/N)
        y = y[0]
        y_complex = tf.complex(y[:N], y[N:])
        x = x[0]

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(np.abs(y_complex))
        axs[0, 0].plot(np.abs(y_pred_complex))
        axs[0, 0].set_title('Intensity of the input pulse (time domain)')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Amplitude')
        axs[0, 1].plot(np.unwrap(np.angle(y_complex)))
        axs[0, 1].plot(np.unwrap(np.angle(y_pred_complex)))
        axs[0, 1].set_title('Phase of the input pulse (time domain)')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Amplitude')
        axs[1, 0].imshow(x)
        axs[1, 0].set_title('SHG-FROG trace of the input pulse')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 1].imshow(pred_trace)
        axs[1, 1].set_title('SHG-FROG trace of the predicted pulse')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Frequency')
        plt.show()
