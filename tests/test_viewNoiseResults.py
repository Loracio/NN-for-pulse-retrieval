"""
In this file we will load a dataset that contains noisy and original traces, and the pulses related to them.

We will pass the noisy traces through a trained model in no noise, and compare the results with the original traces,
comparing the MSE of the traces and the electric field of the pulses.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

import path_helper

from src.io import process_data_tfrecord_noisyTraces
from src.visualization import resultsGUI
from src.models import trace_MSE
from src.utils import fourier_utils


import matplotlib.pyplot as plt
import scienceplots


plt.style.use('science')
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.axisbelow'] = True

def field_error(E_real, E_predicted):
    # Calculate the dot products
    dot_real_real = tf.abs(tf.tensordot(E_real, tf.math.conj(E_real), axes=1))
    dot_predicted_predicted = tf.abs(tf.tensordot(E_predicted, tf.math.conj(E_predicted), axes=1))
    dot_real_predicted = tf.abs(tf.tensordot(E_real, tf.math.conj(E_predicted), axes=1))

    # Calculate the error
    error = tf.acos(dot_real_predicted / tf.sqrt(dot_real_real * dot_predicted_predicted))

    return error


def intensity_error(I_real, I_predicted):
    # Calculate the absolute differences
    absolute_differences = tf.abs(I_real - I_predicted)

    # Calculate the sum of the absolute differences
    sum_absolute_differences = tf.reduce_sum(absolute_differences)

    # Calculate the sum of the absolute real intensities
    sum_absolute_real = tf.reduce_sum(tf.abs(I_real))

    # Calculate the error
    error = tf.divide(sum_absolute_differences, sum_absolute_real)

    return error

if __name__ == '__main__':

    # Define pulse database parameters
    N = 128
    NUMBER_OF_PULSES = 10
    SNR = 15 # Signal to noise ratio in dB
    FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomPulses_N{N}_{SNR}SNR.tfrecords"
    # Handle error if path does not exist
    try:
        with open(FILE_PATH) as f:
            pass
    except FileNotFoundError:
        print("File not found. Please generate the pulse database first.")
        exit()

    # Load the dataset
    train_dataset, test_dataset, train_original_trace_dataset, test_original_trace_dataset, train_retrieved_field_dataset, test_retrieved_field_dataset = process_data_tfrecord_noisyTraces(N, NUMBER_OF_PULSES, FILE_PATH, 0.99, 1)

    model = keras.models.load_model(
        f"./trained_models/CNN/CNN_test_N{N}_normTraces_total_TESTNOISE.tf")

    # Create fourier utils instance to use its utilities
    ft = fourier_utils(N, 1/N)

    # Pass the traces from the train dataset to obtain the predictions
    # Extract inputs from the train_dataset
    noisy_traces = train_dataset.map(lambda x, y: x)

    # convert noisy traces to tensors
    noisy_traces_iterator = noisy_traces.as_numpy_iterator()


    retrieved_pulses_iterator = train_retrieved_field_dataset.as_numpy_iterator()

    # Get pulses from train dataset
    pulses = train_dataset.map(lambda x, y: y)

    pulses_iterator = pulses.as_numpy_iterator()

    train_original_trace_dataset_iterator = train_original_trace_dataset.as_numpy_iterator()

    # Go for each result and plot the traces and the pulses
    
    
    for i in range(int(0.9* NUMBER_OF_PULSES)):

        noisy_traces = next(iter(noisy_traces_iterator))
        noisy_traces = tf.convert_to_tensor(noisy_traces)

        inputs = tf.expand_dims(noisy_traces, -1)

        # Extract the pulses from the train dataset
        pulses = next(iter(pulses_iterator))

        # Convert pulses to tensor
        pulses = tf.convert_to_tensor(pulses)

        # Convert the pulses tensor into a complex tensor with real and imaginary parts
        pulses = tf.complex(pulses[:, :N], pulses[:, N:])

        # Extract the retrieved pulses from the retrieved_field_dataset
        retrieved_pulses = next(iter(retrieved_pulses_iterator))

        # Convert retrieved pulses to tensor
        retrieved_pulses = tf.convert_to_tensor(retrieved_pulses)

        # Compute the retrieved traces
        retrieved_traces = ft.compute_trace(retrieved_pulses)

        retrieved_traces = retrieved_traces / tf.reduce_max(tf.abs(retrieved_traces), axis=[1, 2], keepdims=True)

        # Convert the retrieved pulses tensor into a complex tensor with real and imaginary parts
        retrieved_pulses = tf.complex(retrieved_pulses[:, :N], retrieved_pulses[:, N:])

        # Compute the spectrums
        spectrums = ft.apply_DFT(pulses)

        # Compute the retrieved spectrums
        retrieved_spectrums = ft.apply_DFT(retrieved_pulses)

        # Use the inputs to make predictions
        train_predictions = model.predict(inputs)

        # conver train_predictions to a tensor to use the trace_MSE
        train_predictions = tf.convert_to_tensor(train_predictions)

        # Convert to numpy array
        train_original_trace_dataset = next(iter(train_original_trace_dataset_iterator))

        # Convert numpy array to tensor
        train_original_trace_dataset = tf.convert_to_tensor(train_original_trace_dataset)

        # Concat the train predictions to a complex array, first N elements are the real part, the next N elements are the imaginary part
        predicted_pulses = tf.complex(train_predictions[:, :N], train_predictions[:, N:])

        # Compute predicted spectrums
        predicted_spectrums = ft.apply_DFT(predicted_pulses)

        # # Compute the trace MSE error between predicted and original traces
        train_trace_mse = trace_MSE(N, 1/N)
        # # Compute the traces using utils
        
        predicted_traces = ft.compute_trace(train_predictions)

        # Normalize predicted_traces
        predicted_traces = predicted_traces / tf.reduce_max(tf.abs(predicted_traces), axis=[1, 2], keepdims=True)

        train_trace_mse.update_state(
            train_original_trace_dataset, train_predictions)
        train_trace_mse = train_trace_mse.result().numpy()
        print(f"Train trace MSE: {train_trace_mse:.2e}")

        original_trace, noisy_trace, predicted_trace, pulse, spectrum, predicted_pulse, predicted_spectrum, retrieved_field, retrieved_spectrum, retrieved_trace = train_original_trace_dataset[0], noisy_traces[0], predicted_traces[0], pulses[0], spectrums[0], predicted_pulses[0], predicted_spectrums[0], retrieved_pulses[0], retrieved_spectrums[0], retrieved_traces[0]

        fig, ax = plt.subplots(3, 2, figsize=(12, 8))

        # make the first row of axis only have two
        # ax[0][0].axis('off')


        #! RETRIEVED PULSE COPRA

        # compute error real vs retrieved copra
        error = field_error(pulse, retrieved_field)

        # Plot the pulse: intensity and phase in time domain
        ax[1][1].plot(np.abs(retrieved_field) / np.max(np.abs(retrieved_field)), label="Predicted Intensity", linewidth=4, color='royalblue', alpha=0.9)
        ax[1][1].plot(np.abs(pulse) / np.max(np.abs(pulse)), label="Absolute field", color='black', linewidth=2)
        # show ylabels 0,1 
        ax[1][1].set_yticks([0, 1])
        ax[1][1].set_yticklabels([0, 1])
        ax[1][1].tick_params(axis='y', colors='royalblue')
        
        #! Phase blanking. Where the intensity is below a certain threshold (1e-5), the phase is set to nan
        #! This is done to avoid plotting the phase where the intensity is too low
        phase = np.angle(pulse)
        phase = np.unwrap(phase)
        phase[np.abs(pulse) < 5e-3] = np.nan
        # extract the phase at the peak value of the intensity
        peak = np.argmax(np.abs(pulse))
        # set the phase at the peak value to 0
        phase -= phase[peak]

        predicted_phase = np.angle(retrieved_field)
        predicted_phase = np.unwrap(predicted_phase)
        predicted_phase[np.abs(pulse) < 5e-3] = np.nan
        # extract the phase at the peak value of the intensity
        peak = np.argmax(np.abs(retrieved_field))
        # set the phase at the peak value to 0
        predicted_phase -= predicted_phase[peak]
        
        # create ax twin to plot intensity
        ax_phase = ax[1][1].twinx()
        ax_phase.plot(predicted_phase, '--', label="Predicted Phase", color='red', linewidth=3)
        ax_phase.plot(phase, '--', label="Phase", color='black', linewidth=2)
        # Set ax_phase limits to -2pi to +2pi
        ax_phase.set_ylim(-2*np.pi, 2*np.pi)
        # labels to be -2pi, 0, + 2pi
        ax_phase.set_yticks([-2*np.pi, 0, 2*np.pi])
        ax_phase.set_yticklabels([r'$-2\pi$', r'$0$', r'$+2\pi$'])
        ax_phase.tick_params(axis='y', colors='red')
        ax[1][1].set_xticks([])
        ax[1][1].set_xlabel("Time")
        ax[1][1].set_ylabel("Intensity", color='royalblue')
        ax_phase.set_ylabel("Phase", color='red')
        ax[1][1].set_title(rf"COPRA $\delta$E = {error:.2f}")
        # set the grid for the ax[0][0]
        ax[1][1].grid(False)

        #! PREDICTED PULSE

        # Compute error real vs predicted
        error = field_error(pulse, predicted_pulse)
        

        # Plot the pulse: intensity and phase in time domain
        ax[2][1].plot(np.abs(predicted_pulse) / np.max(np.abs(predicted_pulse)), label="Predicted Intensity", linewidth=4, color='royalblue', alpha=0.9)
        ax[2][1].plot(np.abs(pulse) / np.max(np.abs(pulse)), label="Absolute field", color='black', linewidth=2)
        # show ylabels 0,1
        ax[2][1].set_yticks([0, 1])
        ax[2][1].set_yticklabels([0, 1])
        ax[2][1].tick_params(axis='y', colors='royalblue')
        
        #! Phase blanking. Where the intensity is below a certain threshold (1e-5), the phase is set to nan
        #! This is done to avoid plotting the phase where the intensity is too low
        predicted_phase = np.angle(predicted_pulse)
        predicted_phase = np.unwrap(predicted_phase)
        predicted_phase[np.abs(pulse) < 5e-3] = np.nan
        # extract the phase at the peak value of the intensity
        peak = np.argmax(np.abs(predicted_pulse))
        # set the phase at the peak value to 0
        predicted_phase -= predicted_phase[peak]
        
        # create ax twin to plot intensity
        ax_phase = ax[2][1].twinx()
        ax_phase.plot(predicted_phase, '--', label="Predicted Phase", color='red', linewidth=3)
        ax_phase.plot(phase, '--', label="Phase", color='black', linewidth=2)
        ax_phase.set_ylim(-2*np.pi, 2*np.pi)
        ax_phase.set_yticks([-2*np.pi, 0, 2*np.pi])
        ax_phase.set_yticklabels([r'$-2\pi$', r'$0$', r'$+2\pi$'])
        ax_phase.tick_params(axis='y', colors='red')
        ax[2][1].set_xticks([])
        ax[2][1].set_xlabel("Time")
        ax[2][1].set_ylabel("Intensity", color='royalblue')
        ax_phase.set_ylabel("Phase", color='red')
        ax[2][1].set_title(rf"NN $\delta$E = {error:.2f}")
        # set the grid for the ax[0][0]
        ax[2][1].grid(False)



        #! ORIGINAL TRACE
        # Plot the traces
        ax[0][1].imshow(original_trace, cmap='nipy_spectral', aspect='auto')
        ax[0][1].set_title("Original trace")
        ax[0][1].grid(False)
        # Unset xticks and yticks
        ax[0][1].set_xticks([])
        ax[0][1].set_yticks([])
        # Set xlabel and ylabel to "Time" and "Frequency"
        ax[0][1].set_xlabel("Frequency")
        ax[0][1].set_ylabel("Time")

        #! NOISY TRACE

        ax[0][0].imshow(noisy_trace, cmap='nipy_spectral', aspect='auto')
        ax[0][0].set_title("Noisy trace")
        ax[0][0].grid(False)
        # Unset xticks and yticks
        ax[0][0].set_xticks([])
        ax[0][0].set_yticks([])
        # Set xlabel and ylabel to "Time" and "Frequency"
        ax[0][0].set_xlabel("Frequency")
        ax[0][0].set_ylabel("Time")
        


        #! COPRA RETRIEVED TRACE
        # compute trace error real vs retrieved copra
        Ierror = intensity_error(original_trace, retrieved_trace)

        ax[1][0].imshow(retrieved_trace, cmap='nipy_spectral', aspect='auto')
        ax[1][0].set_title(rf"COPRA retrieved trace $\delta$I = {Ierror:.2f}")
        ax[1][0].grid(False)
        # Unset xticks and yticks
        ax[1][0].set_xticks([])
        ax[1][0].set_yticks([])
        # Set xlabel and ylabel to "Time" and "Frequency"
        ax[1][0].set_xlabel("Frequency")
        ax[1][0].set_ylabel("Time")


        #! NN PREDICTED TRACE
        # Compute trace error real vs predicted
        Ierror = intensity_error(original_trace, predicted_trace)

        ax[2][0].imshow(predicted_trace, cmap='nipy_spectral', aspect='auto')
        ax[2][0].set_title(rf"NN predicted trace $\delta$I = {Ierror:.2f}")
        ax[2][0].grid(False)
        # Unset xticks and yticks
        ax[2][0].set_xticks([])
        ax[2][0].set_yticks([])
        # Set xlabel and ylabel to "Time" and "Frequency"
        ax[2][0].set_xlabel("Frequency")
        ax[2][0].set_ylabel("Time")

        # Set all aspects to equal
        for i in range(3):
            for j in range(2):
                ax[i][j].set_aspect('auto')

        plt.tight_layout()
        plt.show()