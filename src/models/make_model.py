import tensorflow as tf
from tensorflow import keras

def MLP(input_shape, output_shape, n_hidden_layers, n_neurons_per_layer, activation, dropout=None):
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
    # Only add dropout if dropout is not None
    if dropout is not None:
            dense_layer = keras.layers.Dropout(dropout)(dense_layer)
    for i in range(n_hidden_layers - 1):
        dense_layer = keras.layers.Dense(n_neurons_per_layer, activation=activation)(dense_layer)
        # Only add dropout if dropout is not None
        if dropout is not None:
            dense_layer = keras.layers.Dropout(dropout)(dense_layer)
    outputs = keras.layers.Dense(output_shape, name="output")(dense_layer)

    return keras.Model(inputs=inputs, outputs=outputs)

def bottleneck_MLP(input_shape, output_shape, n_hidden_layers, n_neurons_first_layer, reduction_factor, activation, dropout=None):
    """
    This function creates a MLP model with the specified parameters.
    The number of neurons in each layer is reduced by a factor of reduction_factor, hence the name.

    Args:
        input_shape (tuple): Shape of the input layer
        output_shape (tuple): Shape of the output layer
        n_hidden_layers (int): Number of hidden layers
        n_neurons_first_layer (int): Number of neurons in the first hidden layer
        reduction_factor (int): Reduction factor for the number of neurons in each hidden layer
        activation (str): Activation function to use in hidden layers
    """

    inputs = keras.Input(shape=input_shape, name="input")
    flatten_layer = keras.layers.Flatten()(inputs)
    # Add the hidden layers given by the arguments
    dense_layer = keras.layers.Dense(n_neurons_first_layer, activation=activation)(flatten_layer)
    # Only add dropout if dropout is not None
    if dropout is not None:
            dense_layer = keras.layers.Dropout(dropout)(dense_layer)
    for i in range(n_hidden_layers - 1):
        # Reduction factor is applied to the number of neurons in each layer
        dense_layer = keras.layers.Dense(n_neurons_first_layer // (reduction_factor * (i + 1)), activation=activation)(dense_layer)
        # Only add dropout if dropout is not None
        if dropout is not None:
            dense_layer = keras.layers.Dropout(dropout)(dense_layer)
    outputs = keras.layers.Dense(output_shape, name="output")(dense_layer)

    return keras.Model(inputs=inputs, outputs=outputs)