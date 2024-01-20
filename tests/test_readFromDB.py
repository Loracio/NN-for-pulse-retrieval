import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import path_helper
from src.io import load_data, load_and_norm_data

if __name__ == '__main__':
    N = 64
    NUMBER_OF_PULSES = 1000
    FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"

    # Load the dataset into a tf.data.Dataset object
    pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)

    # Take one random element of the dataset and plot the pulse and the SHG-FROG trace
    for tbp, train, target in pulse_dataset.take(1):
        # Plot the pulse
        # The first N elements are the real part of the pulse, the last N are the imaginary part
        # We take them and create a complex array

        pulse = tf.complex(target[:N], target[N:])
        # Plot the intensity and phase
        plt.plot(np.abs(pulse))
        plt.plot(np.angle(pulse), linestyle='--')
        plt.show()

        # Plot the SHG-FROG trace
        plt.imshow(tf.reshape(train, (N, N)))
        plt.show()

        # Print the TBP
        print(tbp)

    # Go through the whole dataset and check that the shapes are correct
    # tbp should be a scalar
    # train should be a NxN matrix
    # target should be a 2N-dimensional vector
    for tbp, train, target in pulse_dataset:
        if tbp.shape != ():
            print("Error tbp shape")
        if train.shape != (N, N):
            print("Error train shape")
        if target.shape != (2*N,):
            print("Error target shape")