import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import path_helper

from src.visualization import resultsGUI
from src.io import load_and_norm_data, process_data


if __name__ == '__main__':
    # Define pulse database parameters
    N = 128
    Δt = 1 / N
    NUMBER_OF_PULSES = 500
    FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"
    # Handle error if path does not exist
    try:
        with open(FILE_PATH) as f:
            pass
    except FileNotFoundError:
        print("File not found. Please generate the pulse database first.")
        exit()

    # Load and process the pulse database
    pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)
    train_dataset, test_dataset = process_data(N, NUMBER_OF_PULSES, pulse_dataset,
                                              0.8, 32)

    # Load the trained model
    # model = keras.models.load_model(f"./trained_models/FCNN/MLP_test2.h5") #! Best MSE sweep result
    # model = keras.models.load_model(f"./trained_models/FCNN/bottleneck_MLP.h5") #! Bottleneck with MSE
    # model = keras.models.load_model(f"./trained_models/FCNN/MLP_test1.h5") #! Custom loss example
    # model = keras.models.load_model(f"./trained_models/CNN/CNN_test1.h5") #! Custom loss example with CNN + GPU!
    # model = keras.models.load_model(f"./trained_models/FCNN/bottleneck_MLP_custom_losstest1.h5") #! Custom loss example with CNN + GPU!
    model = keras.models.load_model(f"./trained_models/CNN/CNN_test_N128.h5") #! N=128
    

    # Create the GUI
    results = resultsGUI(model, train_dataset, int(0.8 * NUMBER_OF_PULSES), N, Δt, norm_predictions=True)
    results.plot()

    plt.show()