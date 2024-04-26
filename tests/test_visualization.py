import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import path_helper

from src.visualization import resultsGUI
from src.io import load_and_norm_data, process_data, process_data_tfrecord


if __name__ == '__main__':
    # Define pulse database parameters
    N = 128
    Δt = 1 / N
    NUMBER_OF_PULSES = 1000
    FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomNormalizedPulses_N{N}.tfrecords"
    # Handle error if path does not exist
    try:
        with open(FILE_PATH) as f:
            pass
    except FileNotFoundError:
        print("File not found. Please generate the pulse database first.")
        exit()

    # Load and process the pulse database
    # pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)
    # train_dataset, test_dataset = process_data(N, NUMBER_OF_PULSES, pulse_dataset,
    #                                           0.8, 32)

    train_dataset, test_dataset = process_data_tfrecord(N, NUMBER_OF_PULSES, FILE_PATH, 0.1, 10, add_noise=False, noise_level=0.01, mask=True, mask_tolerance=1e-3)

    # Liberate memory from test_dataset
    del test_dataset

    # Load the trained model
    # model = keras.models.load_model(f"./trained_models/FCNN/MLP_test2.h5") #! Best MSE sweep result
    # model = keras.models.load_model(f"./trained_models/FCNN/bottleneck_MLP.h5") #! Bottleneck with MSE
    # model = keras.models.load_model(f"./trained_models/FCNN/MLP_test1.h5") #! Custom loss example
    # model = keras.models.load_model(f"./trained_models/CNN/CNN_test1.h5") #! Custom loss example with CNN + GPU!
    # model = keras.models.load_model(f"./trained_models/FCNN/bottleneck_MLP_custom_losstest1.h5") #! Custom loss example with CNN + GPU!
    # model = keras.models.load_model(f"./trained_models/CNN/CNN_test_N128.h5") #! N=128
    model = keras.models.load_model(f"./trained_models/CNN/CNN_test_N{N}_normTraces_combinedTraining_33.tf") #! N=128 with custom loss
    

    # Create the GUI
    results = resultsGUI(model, train_dataset, int(0.1 * NUMBER_OF_PULSES), N, Δt, norm_predictions=True, phase_blanking=True, phase_blanking_threshold=1e-5)
    results.plot()

    plt.show()