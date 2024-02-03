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


def CNN(input_shape, output_shape, n_conv_layers, n_filters_per_layer, reduce_filter_factor, kernel_size, pool, pool_size, conv_activation, n_dense_layers, n_neurons_per_layer, reduce_dense_factor, dense_activation, dropout=None):
    """
    This function creates a CNN model with the specified parameters.

    Args:
        input_shape (tuple): Shape of the input layer
        output_shape (tuple): Shape of the output layer
        n_conv_layers (int): Number of convolutional layers
        n_filters_per_layer (int): Number of filters per layer
        reduce_filter_factor (int): Reduction factor for the number of filters in each layer
        kernel_size (tuple): Kernel size
        pool (bool): Whether to add a pooling layer after each convolutional layer
        pool_size (tuple): Pool size
        conv_activation (str): Activation function for the convolutional layers
        n_dense_layers (int): Number of dense layers
        n_neurons_per_layer (int): Number of neurons per dense layer
        reduce_dense_factor (int): Reduction factor for the number of neurons in each layer in the dense layers
        dense_activation (str): Activation function for the dense layers
        dropout (float): Dropout rate, if None, no dropout is used
    """
    inputs = keras.Input(shape=input_shape, name="input")
    # Add the convolutional layers given by the arguments
    conv_layer = keras.layers.Conv2D(n_filters_per_layer, kernel_size, activation=conv_activation)(inputs)
    if pool:
        conv_layer = keras.layers.MaxPooling2D(pool_size=pool_size)(conv_layer)
    for i in range(n_conv_layers - 1):
        # Reduction factor is applied to the number of filters in each layer
        conv_layer = keras.layers.Conv2D(n_filters_per_layer // (reduce_filter_factor * (i + 1)), kernel_size, activation=conv_activation)(conv_layer)
        if pool:
            conv_layer = keras.layers.MaxPooling2D(pool_size=pool_size)(conv_layer)
    flatten_layer = keras.layers.Flatten()(conv_layer)
    # Add the dense layers given by the arguments
    dense_layer = keras.layers.Dense(n_neurons_per_layer, activation=dense_activation)(flatten_layer)
    # Only add dropout if dropout is not None
    if dropout is not None:
            dense_layer = keras.layers.Dropout(dropout)(dense_layer)
    for i in range(n_dense_layers - 1):
        # Reduction factor is applied to the number of neurons in each layer
        dense_layer = keras.layers.Dense(n_neurons_per_layer // (reduce_dense_factor * (i + 1)), activation=dense_activation)(dense_layer)
        # Only add dropout if dropout is not None
        if dropout is not None:
            dense_layer = keras.layers.Dropout(dropout)(dense_layer)
    outputs = keras.layers.Dense(output_shape, name="output")(dense_layer)

    return keras.Model(inputs=inputs, outputs=outputs)