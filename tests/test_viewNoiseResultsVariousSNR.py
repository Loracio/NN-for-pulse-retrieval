"""
! Very messy code. I will clean it up later.

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
plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.axisbelow'] = True

# def field_error(E_real, E_predicted):
#     # Calculate the dot products
#     dot_real_real = tf.abs(tf.tensordot(E_real, tf.math.conj(E_real), axes=1))
#     dot_predicted_predicted = tf.abs(tf.tensordot(E_predicted, tf.math.conj(E_predicted), axes=1))
#     dot_real_predicted = tf.abs(tf.tensordot(E_real, tf.math.conj(E_predicted), axes=1))

#     # Calculate the error
#     error = tf.acos(dot_real_predicted / tf.sqrt(dot_real_real * dot_predicted_predicted))

#     return error

def field_error(E_real, E_predicted):
    # Convert E_real and E_predicted to numpy arrays
    E_real = E_real.numpy()
    E_predicted = E_predicted.numpy()

    # Normalize E_predicted
    E_predicted = E_predicted / np.max(np.abs(E_predicted))

    # Compute the element-wise difference in absolute value
    absolute_differences = np.abs(E_real - E_predicted)
    # Divide by the length of the array
    error = np.sum(absolute_differences) / len(E_real)

    return error


def trace_error(I_real, I_predicted):
    # Calculate the absolute differences
    absolute_differences = tf.abs(I_real - I_predicted)

    # Calculate the sum of the absolute differences
    sum_absolute_differences = tf.reduce_sum(absolute_differences)

    # Calculate the sum of the absolute real intensities
    sum_absolute_real = tf.reduce_sum(tf.abs(I_real))

    # Calculate the error
    error = tf.divide(sum_absolute_differences, sum_absolute_real)

    return error

# def trace_error(I_real, I_predicted):
#     # Convert to numpy arrays
#     I_real = I_real.numpy()
#     I_predicted = I_predicted.numpy()

#     error = np.sqrt(np.sum((I_real - I_predicted)**2) / len(I_real))

#     return error

if __name__ == '__main__':

    # Define pulse database parameters
    N = 128
    NUMBER_OF_PULSES = 1000
    SNR_list = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] # Signal to noise ratio in dB
    # SNR_list = [50] # Signal to noise ratio in dB

    meanTraceErrorsRetrievedSNR = []
    meanTraceErrorsPredictedSNR = []
    meanFieldErrorsRetrievedSNR = []
    meanFieldErrorsPredictedSNR = []

    stdTraceErrorsRetrievedSNR = []
    stdTraceErrorsPredictedSNR = []
    stdFieldErrorsRetrievedSNR = []
    stdFieldErrorsPredictedSNR = []

    traceMSEs = []

    model = keras.models.load_model(
            f"./trained_models/CNN/CNN_test_N{N}_normTraces_combinedTraining_BIGDB.tf")

    # Create fourier utils instance to use its utilities
    ft = fourier_utils(N, 1/N)
    train_trace_mse = trace_MSE(N, 1/N)

    for SNR in SNR_list:
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
        
        traceRetrievedErrors = []
        tracePredictionErrors = []

        fieldRetrievedErrors = []
        fieldPredictionErrors = []

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
            
            predicted_traces = ft.compute_trace(train_predictions)

            # Normalize predicted_traces
            predicted_traces = predicted_traces / tf.reduce_max(tf.abs(predicted_traces), axis=[1, 2], keepdims=True)

            train_trace_mse.update_state(
                train_original_trace_dataset, train_predictions)

            original_trace, noisy_trace, predicted_trace, pulse, predicted_pulse, retrieved_field, retrieved_trace = train_original_trace_dataset[0], noisy_traces[0], predicted_traces[0], pulses[0], predicted_pulses[0], retrieved_pulses[0], retrieved_traces[0]

            # normalize pulse
            pulse =pulse / tf.cast( tf.reduce_max(tf.abs(pulse)), tf.complex64)

            # normalize retrieved_field
            retrieved_field = retrieved_field / tf.cast( tf.reduce_max(tf.abs(retrieved_field)), tf.complex64)

            # normalize predicted_pulse
            predicted_pulse = predicted_pulse /tf.cast( tf.reduce_max(tf.abs(predicted_pulse)), tf.complex64)


            # Error computation
            retrieved_error = field_error(pulse, retrieved_field)

            flipped_conj_retrieved_field = tf.reverse(tf.math.conj(retrieved_field), axis=[0])
            # move the maximum intensity to the center of the array
            max_index = tf.argmax(tf.abs(retrieved_field))
            flipped_conj_retrieved_field = tf.roll(flipped_conj_retrieved_field, shift=N//2 - max_index + 1, axis=0)

            retrieved_error = tf.minimum(retrieved_error, field_error(pulse, flipped_conj_retrieved_field))

            # compute error real vs retrieved copra
            fieldRetrievedErrors.append(retrieved_error)


            # Compute error real vs predicted

            predicted_error = field_error(pulse, predicted_pulse)

            flipped_conj_predicted_pulse = tf.reverse(tf.math.conj(predicted_pulse), axis=[0])
            # move the maximum intensity to the center of the array
            max_index = tf.argmax(tf.abs(predicted_pulse))
            flipped_conj_predicted_pulse = tf.roll(flipped_conj_predicted_pulse, shift=N//2 - max_index + 1, axis=0)

            predicted_error = tf.minimum(predicted_error, field_error(pulse, flipped_conj_predicted_pulse))

            fieldPredictionErrors.append(predicted_error)
            
            # compute trace error real vs retrieved copra
            traceRetrievedErrors.append(trace_error(original_trace, retrieved_trace))
            # Compute trace error real vs predicted
            tracePredictionErrors.append(trace_error(original_trace, predicted_trace))

        traceMSEs.append(train_trace_mse.result().numpy())

        # reset train_trace_mse
        train_trace_mse.reset_states()

        # Compute mean and append to the list of Retrieved means (field and trace)
        meanFieldErrorsRetrievedSNR.append(np.mean(fieldRetrievedErrors))
        meanTraceErrorsRetrievedSNR.append(np.mean(traceRetrievedErrors))

        # Compute mean and append to the list of Predicted means (field and trace)
        meanFieldErrorsPredictedSNR.append(np.mean(fieldPredictionErrors))
        meanTraceErrorsPredictedSNR.append(np.mean(tracePredictionErrors))

        # Compute std and append to the list of Retrieved std (field and trace)
        stdFieldErrorsRetrievedSNR.append(np.std(fieldRetrievedErrors))
        stdTraceErrorsRetrievedSNR.append(np.std(traceRetrievedErrors))

        # Compute std and append to the list of Predicted std (field and trace)
        stdFieldErrorsPredictedSNR.append(np.std(fieldPredictionErrors))
        stdTraceErrorsPredictedSNR.append(np.std(tracePredictionErrors))

     # Save results of mean and std to a file
    np.save(f"./meanFieldErrorsRetrievedCOPRASNR_N{N}.npy", meanFieldErrorsRetrievedSNR)
    np.save(f"./meanTraceErrorsRetrievedCOPRASNR_N{N}.npy", meanTraceErrorsRetrievedSNR)
    np.save(f"./meanFieldErrorsPredictedSNR_N{N}.npy", meanFieldErrorsPredictedSNR)
    np.save(f"./meanTraceErrorsPredictedSNR_N{N}.npy", meanTraceErrorsPredictedSNR)

    np.save(f"./stdFieldErrorsRetrievedCOPRASNR_N{N}.npy", stdFieldErrorsRetrievedSNR)
    np.save(f"./stdTraceErrorsRetrievedCOPRASNR_N{N}.npy", stdTraceErrorsRetrievedSNR)
    np.save(f"./stdFieldErrorsPredictedSNR_N{N}.npy", stdFieldErrorsPredictedSNR)
    np.save(f"./stdTraceErrorsPredictedSNR_N{N}.npy", stdTraceErrorsPredictedSNR)

    # Load mean and std results of NN
    meanFieldErrorsPredictedSNR = np.load(f"./meanFieldErrorsPredictedSNR_N{N}.npy")
    meanTraceErrorsPredictedSNR = np.load(f"./meanTraceErrorsPredictedSNR_N{N}.npy")
    stdFieldErrorsPredictedSNR = np.load(f"./stdFieldErrorsPredictedSNR_N{N}.npy")
    stdTraceErrorsPredictedSNR = np.load(f"./stdTraceErrorsPredictedSNR_N{N}.npy")

    # Load mean and std results of COPRA
    meanFieldErrorsRetrievedCOPRASNR = np.load(f"./meanFieldErrorsRetrievedCOPRASNR_N{N}.npy")
    meanTraceErrorsRetrievedCOPRASNR = np.load(f"./meanTraceErrorsRetrievedCOPRASNR_N{N}.npy")
    stdFieldErrorsRetrievedCOPRASNR = np.load(f"./stdFieldErrorsRetrievedCOPRASNR_N{N}.npy")
    stdTraceErrorsRetrievedCOPRASNR = np.load(f"./stdTraceErrorsRetrievedCOPRASNR_N{N}.npy")


    # # Load mean and std results of PIE
    # meanFieldErrorsRetrievedPIESNR = np.load(f"./meanFieldErrorsRetrievedPIESNR_N{N}.npy")
    # meanTraceErrorsRetrievedPIESNR = np.load(f"./meanTraceErrorsRetrievedPIESNR_N{N}.npy")
    # stdFieldErrorsRetrievedPIESNR = np.load(f"./stdFieldErrorsRetrievedPIESNR_N{N}.npy")
    # stdTraceErrorsRetrievedPIESNR = np.load(f"./stdTraceErrorsRetrievedPIESNR_N{N}.npy")

    # # # Load mean and std results of GPA
    # # meanFieldErrorsRetrievedGPASNR = np.load(f"./meanFieldErrorsRetrievedGPASNR_N{N}.npy")
    # # meanTraceErrorsRetrievedGPASNR = np.load(f"./meanTraceErrorsRetrievedGPASNR_N{N}.npy")
    # # stdFieldErrorsRetrievedGPASNR = np.load(f"./stdFieldErrorsRetrievedGPASNR_N{N}.npy")
    # # stdTraceErrorsRetrievedGPASNR = np.load(f"./stdTraceErrorsRetrievedGPASNR_N{N}.npy")



    # Plot the mean errors of the traces and the electric field for each SNR (and its std)
    fig, ax = plt.subplots(1, 2)

    ax[0].errorbar(SNR_list, meanTraceErrorsRetrievedCOPRASNR, yerr=stdTraceErrorsRetrievedCOPRASNR, label="COPRA", fmt='o-', capsize=5)
    # ax[0].errorbar(SNR_list, meanTraceErrorsRetrievedPIESNR, yerr=stdTraceErrorsRetrievedPIESNR, label="PIE", fmt='o-', capsize=5)
    # ax[0].errorbar(SNR_list, meanTraceErrorsRetrievedGPASNR, yerr=stdTraceErrorsRetrievedGPASNR, label="GPA", fmt='o-', capsize=5)
    ax[0].errorbar(SNR_list, meanTraceErrorsPredictedSNR, yerr=stdTraceErrorsPredictedSNR, label="Neural Network", fmt='o-', capsize=5)
    ax[0].set_title("Trace Error")
    ax[0].set_xlabel("SNR (dB)")
    ax[0].set_ylabel(r"$\delta$I")
    ax[0].legend()


    ax[1].errorbar(SNR_list, meanFieldErrorsRetrievedCOPRASNR, yerr=stdFieldErrorsRetrievedCOPRASNR, label="COPRA", fmt='o-', capsize=5)
    # ax[1].errorbar(SNR_list, meanFieldErrorsRetrievedPIESNR, yerr=stdFieldErrorsRetrievedPIESNR, label="PIE", fmt='o-', capsize=5)
    # ax[1].errorbar(SNR_list, meanFieldErrorsRetrievedGPASNR, yerr=stdFieldErrorsRetrievedGPASNR, label="GPA", fmt='o-', capsize=5)
    ax[1].errorbar(SNR_list, meanFieldErrorsPredictedSNR, yerr=stdFieldErrorsPredictedSNR, label="Neural Network", fmt='o-', capsize=5)
    ax[1].set_title("Electric Field Error")
    ax[1].set_xlabel("SNR (dB)")
    ax[1].set_ylabel(r"$\delta$E")
    ax[1].legend()

    plt.tight_layout()
    plt.show()



