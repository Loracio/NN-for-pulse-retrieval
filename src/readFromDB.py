"""
This file contains the functions to read from the database of random pulses, and format the data to be used in the neural network.

The database is a csv file with the following structure:
    TBP, E, Tmn
where:
    TBP: Time between pulses
    E: Pulse in the time domain real part, then imaginary part
    Tmn: SHG-FROG trace of the pulse

E is a 2N-dimensional vector, and Tmn is a NxN matrix. N is given by the user.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def load_data(FILE_PATH, N, NUMBER_OF_PULSES):
    """
    This function preprocesses the data from the database, iterating over it.
    It returns a dataset with the input data (train) and the target data (target)

    Note that we also have the TBP of the pulses in the first column of the db.
    We want to save them in a separate array, so we can use them later.
    """
    # Create a record_defaults with 1 + 2N + N*N elements that are tf.float32
    db_record_defaults = [tf.float32] * (1 + 2*N + N*N)

    # Read the database
    pulse_db = tf.data.experimental.CsvDataset(
        FILE_PATH, record_defaults=db_record_defaults, header=False)

    # Create empty datasets
    tbp_dataset = tf.data.Dataset.from_tensor_slices([])
    train_dataset = tf.data.Dataset.from_tensor_slices([])
    target_dataset = tf.data.Dataset.from_tensor_slices([])

    # Iterate over the database
    for i, pulse in enumerate(pulse_db):
        # Save the TBP in the tbp_dataset
        tbp_dataset = tbp_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.reshape(pulse[0], (1,))))

        # Save the SHG-FROG trace in the train_dataset
        train_dataset = train_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.reshape(pulse[2*N + 1:], (1, N, N))))

        # Save the pulse in the target_dataset
        target_dataset = target_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.reshape(pulse[1:2*N + 1], (1, 2*N))))

    # Create the final dataset
    dataset = tf.data.Dataset.zip((tbp_dataset, train_dataset, target_dataset))

    return dataset


if __name__ == '__main__':
    N = 64
    NUMBER_OF_PULSES = 1000
    FILE_PATH = f"./src/db/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"

    # Load the dataset into a tf.data.Dataset object
    pulse_dataset = load_data(FILE_PATH, N, NUMBER_OF_PULSES)

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