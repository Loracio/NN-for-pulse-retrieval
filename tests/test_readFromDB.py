import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import path_helper
from src.io import process_data_tfrecord

if __name__ == '__main__':
    N = 128
    NUMBER_OF_PULSES = 100
    FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomNormalizedPulses_N{N}.tfrecords"

    # Load the dataset into a tf.data.Dataset object
    # pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)
    train_dataset, test_dataset = process_data_tfrecord(N, NUMBER_OF_PULSES, FILE_PATH, 0.8, 32, norm_traces='individual')

    # Take one random element of the dataset and plot the pulse and the SHG-FROG trace
    for train_batch, target_batch  in train_dataset.take(33):
        for target, train in zip(target_batch, train_batch):
            # Plot the pulse
            # The first N elements are the real part of the pulse, the last N are the imaginary part
            # We take them and create a complex array
            pulse = tf.complex(target[:N], target[N:])
            # Plot the intensity and phase
            plt.plot(np.abs(pulse))
            plt.plot((np.angle(pulse)), linestyle='--')
            # draw a vertical line at x = N/2
            # draw an horizontal line at y = 0
            plt.axvline(x=N/2, color='k', linestyle='--')
            plt.axhline(y=0, color='k', linestyle='--') 
            plt.show()

            # Plot the SHG-FROG trace
            plt.imshow(train)
            # Plot bar
            plt.colorbar()
            plt.show()