import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import path_helper

from src.visualization import resultsGUI
from src.io import load_and_norm_data, process_data, process_data_tfrecord
from src.models import trace_MSE
from src.utils import fourier_utils

import scienceplots


plt.style.use('science')
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = [11.69,8.27]
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.axisbelow'] = True


# def intensity_error(I_real, I_predicted):
#     # Calculate the absolute differences
#     absolute_differences = tf.abs(I_real - I_predicted)

#     # Calculate the sum of the absolute differences
#     sum_absolute_differences = tf.reduce_sum(absolute_differences)

#     # Calculate the sum of the absolute real intensities
#     sum_absolute_real = tf.reduce_sum(tf.abs(I_real))

#     # Calculate the error
#     error = tf.divide(sum_absolute_differences, sum_absolute_real)

#     return error

def  intensity_error(I_real, I_predicted):
    # Convert to numpy arrays
    # I_real = I_real.numpy()
    # I_predicted = I_predicted.numpy()

    # Normalise both arrays
    I_real = I_real / np.max(I_real)
    I_predicted = I_predicted / np.max(I_predicted)

    error = np.sqrt(np.sum((I_real - I_predicted)**2) / len(I_real))

    return error


if __name__ == '__main__':
    # Define pulse database parameters
    N = 128
    Δt = 1 / N
    
    PATH_1 = "/home/vloras/Escritorio/github/ultrafast-pulse-retrieval/tests/expResults/compressed_pulse_900mA_trace.csv"
    # PATH_1 = "/home/vloras/Escritorio/github/ultrafast-pulse-retrieval/tests/expResults/compressed_pulse_2100mA_trace.csv"
    # PATH_1 = "/home/vloras/Escritorio/github/ultrafast-pulse-retrieval/tests/expResults/non_compressed_pulse_900mA_trace.csv"
    # PATH_1 = "/home/vloras/Escritorio/github/ultrafast-pulse-retrieval/tests/expResults/non_compressed_pulse_2100mA_trace.csv"


    # Read the csv, each row contains N values and there are N rows, save to 2D array
    exp_trace_1 = np.loadtxt(PATH_1, delimiter=',')

    # Normalize the traces
    exp_trace_1 = exp_trace_1 / np.max(exp_trace_1)


    # transpose 
    # exp_trace_1 = np.transpose(exp_trace_1)
    # exp_trace_2 = np.transpose(exp_trace_2)
    # exp_trace_3 = np.transpose(exp_trace_3)
    # exp_trace_4 = np.transpose(exp_trace_4)

    # Prepare the traces to pass them to the model
    exp_trace_1n = np.expand_dims(exp_trace_1, axis=0)


    # Load the trained model
    model = keras.models.load_model(f"./trained_models/CNN/CNN_test_N{N}_normTraces_combinedTraining_BIGDB.tf") #! N=128 with custom loss

    # Predict the fields
    pred_field_1 = model.predict(exp_trace_1n)


    # The predictions have the N real values and N complex values of the electric field, create an array with the complex values
    pred_field_1_concat = pred_field_1[0, :N] + 1j * pred_field_1[0, N:]


    # Plot the results for each pulse. Retrieved field, retrieved trace, measured trace, difference in a (2,2) subplot
    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Compressed pulse 900mA")

    I_field = np.abs(pred_field_1_concat)
    # normalize the field
    I_field = I_field / np.max(I_field)
    
    phase = np.angle(pred_field_1_concat)
    # apply phase blanking where intensity < 1e-1
    phase[I_field < 5e-2] = np.nan


    ax[0, 0].plot(I_field, color='royalblue')
    ax[0, 0].set_title("Retrieved field")
    ax[0, 0].set_ylabel("Amplitude (a.u.)")
    ax[0, 0].set_xlabel("Time (a.u.)")
    # Plot phase
    # Create twin axis
    ax_phase = ax[0, 0].twinx()
    ax_phase.plot(phase, '--', color='red')
    # Set the limits of the phase axis
    ax_phase.set_ylim(-2 * np.pi, 2 * np.pi)
    # Labels as -2π, 0, +2π in latex style
    ax_phase.set_yticks([-2 * np.pi, 0, 2 * np.pi])
    ax_phase.set_yticklabels([r"$-2\pi$", "0", r"$2\pi$"])
    ax_phase.set_ylabel("Phase (rad)")
    # Set color of ticks and labels to red
    ax_phase.yaxis.label.set_color('red')
    ax_phase.tick_params(axis='y', colors='red')
    ax_phase.spines['right'].set_color('red')
    # Set color of original axis and labels to royalblue
    ax[0, 0].yaxis.label.set_color('royalblue')
    ax[0, 0].tick_params(axis='y', colors='royalblue')
    ax[0, 0].spines['left'].set_color('royalblue')

    # Compute the predicted trace
    ft = fourier_utils(N, 1/N)

    # Compute the retrieved traces
    pred_field_1 = tf.convert_to_tensor(pred_field_1)
    # create an extra dimension for the batch for compatibility
    predicted_traces = ft.compute_trace(pred_field_1)

    predicted_traces = predicted_traces / tf.reduce_max(tf.abs(predicted_traces), axis=[1, 2], keepdims=True)

    predicted_traces = predicted_traces[0, :, :]
    # cast to float64
    predicted_traces = tf.cast(predicted_traces, tf.float64)

    # Plot the retrieved trace

    Ierror = intensity_error(exp_trace_1n, predicted_traces)

    ax[1][0].imshow(predicted_traces, cmap='nipy_spectral', aspect='auto')
    ax[1][0].set_title(rf"NN predicted trace R = {Ierror:.2f}")
    ax[1][0].grid(False)
    # Unset xticks and yticks
    ax[1][0].set_xticks([])
    ax[1][0].set_yticks([])
    # Set xlabel and ylabel to "Time" and "Frequency"
    ax[1][0].set_xlabel("Frequency")
    ax[1][0].set_ylabel("Time")
    # Add colorbar
    fig.colorbar(ax[1][0].imshow(predicted_traces, cmap='nipy_spectral', aspect='auto'), ax=ax[1][0])


    # Plot the traces
    ax[1][1].imshow(exp_trace_1, cmap='nipy_spectral', aspect='auto')
    ax[1][1].set_title("Measured trace")
    ax[1][1].grid(False)
    # Unset xticks and yticks
    ax[1][1].set_xticks([])
    ax[1][1].set_yticks([])
    # Set xlabel and ylabel to "Time" and "Frequency"
    ax[1][1].set_xlabel("Frequency")
    ax[1][1].set_ylabel("Time")
    # Add colorbar
    fig.colorbar(ax[1][1].imshow(exp_trace_1, cmap='nipy_spectral', aspect='auto'), ax=ax[1][1])

    # Add the absolute difference pixel to pixel
    diff = tf.abs(exp_trace_1 - predicted_traces)
    ax[0][1].imshow(diff, cmap='nipy_spectral', aspect='auto')
    ax[0][1].set_title("Difference")
    ax[0][1].grid(False)
    # Unset xticks and yticks
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])
    # Set xlabel and ylabel to "Time" and "Frequency"
    ax[0][1].set_xlabel("Frequency")
    ax[0][1].set_ylabel("Time")
    # Add colorbar
    fig.colorbar(ax[0][1].imshow(diff, cmap='nipy_spectral', aspect='auto'), ax=ax[0][1])

    plt.show()