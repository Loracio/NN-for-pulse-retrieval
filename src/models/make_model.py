import tensorflow as tf
from tensorflow import keras

def MLP(input_shape, output_shape, n_hidden_layers, n_neurons_per_layer, activation):
    """
    This function creates a MLP model with the specified parameters.

    Args:
        input_shape (tuple): Shape of the input layer
        output_shape (tuple): Shape of the output layer
        n_hidden_layers (int): Number of hidden layers
        n_neurons_per_layer (int): Number of neurons per hidden layer
        activation (str): Activation function to use in hidden layers
    """

    inputs = keras.Input(shape=input_shape, name="input")
    flatten_layer = keras.layers.Flatten()(inputs)
    # Add the hidden layers given by the arguments
    dense_layer = keras.layers.Dense(n_neurons_per_layer, activation=activation)(flatten_layer)
    for i in range(number_hidden_layers - 1):
        dense_layer = keras.layers.Dense(n_neurons_per_layer, activation=activation)(dense_layer)
    outputs = keras.layers.Dense(output_shape, name="output")(dense_layer)

    return keras.Model(inputs=inputs, outputs=outputs)