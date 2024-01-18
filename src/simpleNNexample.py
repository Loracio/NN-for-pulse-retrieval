"""
In this script we build a simple NN with 1 hidden layer and 1 output layer
to predict the Electric field of a pulse in the time domain from its SHG-FROG trace.

The input data is the SHG-FROG trace of the pulse (NxN vector), and the target data is the
Electric field of the pulse in the time domain (2N-dimensional vector containing
the real and imaginary parts of the pulse).
"""

import numpy as np
import matplotlib.pyplot as plt
from readFromDB import load_data

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    N = 64
    NUMBER_OF_PULSES = 1000
    FILE_PATH = f"./src/db/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"

    # Load the data
    pulse_dataset = load_data(FILE_PATH, N, NUMBER_OF_PULSES)

    # Split the dataset into train and test
    train_dataset = pulse_dataset.take(int(0.8 * NUMBER_OF_PULSES))
    test_dataset = pulse_dataset.skip(int(0.8 * NUMBER_OF_PULSES))

    # Shuffle the train dataset
    train_dataset = train_dataset.shuffle(buffer_size=NUMBER_OF_PULSES)

    # Batch the datasets
    BATCH_SIZE = 32
    train_dataset = train_dataset.batch(BATCH_SIZE)
    
    # We also need to batch the test data. The reason is that we need to
    # flatten the input data, and the batch() method only works with
    # datasets with the same shape. So we need to batch the test data
    # in order to have the same shape as the train data.
    test_dataset = test_dataset.batch(BATCH_SIZE)

    input_shape = (N, N)
    hidden_layer_neurons = 2 * N
    target_shape = int(2 * N)

    # Build the model
    # We have to flatten the input data, since the Dense layer only accepts 1D data
    input_tensor = Input(shape=input_shape)
    flatten_layer = Flatten()(input_tensor)
    dense_layer = Dense(hidden_layer_neurons, activation='relu')(flatten_layer)
    outputs = Dense(target_shape)(dense_layer)

    model = Model(inputs=input_tensor, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics='mse')

    
    # Extract the input and target data from the dataset. We don't need the TBP
    train_dataset_NN = train_dataset.map(lambda x, y, z: (y, z))
    test_dataset_NN = test_dataset.map(lambda x, y, z: (y, z))

    # Train the model
    EPOCHS = 100
    history = model.fit(train_dataset_NN, validation_data=test_dataset_NN,
                        epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate the model
    loss, mse = model.evaluate(test_dataset_NN, verbose=2)

    # Make predictions
    predictions = model.predict(test_dataset_NN)

    # Plot the results
    plt.figure()
    plt.plot(history.history['loss'], label='loss')

    plt.show()
