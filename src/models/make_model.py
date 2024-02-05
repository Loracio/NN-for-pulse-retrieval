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

def MultiResBlock(input_layer, filters):
    """
    This function creates a MultiRes block with the specified parameters.
    A MultiRes block is a block that applies convolutional layers with different filter sizes and concatenates the outputs.
    In this case the filter sizes are 11, 7, 5 and 3; and the number of filters is the same for all the convolutional layers.

    Args:
        input_layer (keras.layers.Layer): Input layer
        filters (int): Number of filters to use in the convolutional layers

    Returns:
        keras.layers.Layer: Output layer of the MultiRes block
    """
    # Define the different filter sizes
    filter_sizes = [11, 7, 5, 3]

    # Apply convolutional layers with different filter sizes
    conv_layers = [keras.layers.Conv2D(filters, kernel_size=size, strides=1, padding='same', activation='relu')(input_layer) for size in filter_sizes]

    # Concatenate the outputs along the channel dimension
    output_layer = keras.layers.Concatenate()(conv_layers)

    return output_layer

def MultiResNet():
    """
    This function creates a MultiResNet model with the specified parameters.
    A MultiResNet is a CNN that uses MultiRes blocks instead of standard convolutional layers.


    Returns:
        keras.Model: MultiResNet model
    """
    # Define the input layer
    input_layer = keras.layers.Input(shape=(64, 64, 1))

    # Apply MultiRes blocks followed by standard convolutional layers
    x = MultiResBlock(input_layer, 32)
    x = keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = MultiResBlock(x, 64)
    x = keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = MultiResBlock(x, 128)
    x = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu')(x)

    # Flatten the output and apply fully connected layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    output_layer = keras.layers.Dense(128, activation='relu')(x)

    # Define the model
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

# DenseNet implementation

def dense_block(x, blocks, growth_rate):
    """
    Create a dense block with the specified number of blocks and growth rate.
    A dense block is a block that applies a series of convolutional layers and concatenates the outputs.

    Args:
        x (keras.layers.Layer): Input layer
        blocks (int): Number of blocks
        growth_rate (int): Growth rate

    Returns:
        keras.layers.Layer: Output layer of the dense block
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate)
    return x

def transition_block(x, reduction):
    """
    Create a transition block with the specified reduction factor.
    A transition block is a block that applies a series of convolutional layers and pooling layers.

    Args:
        x (keras.layers.Layer): Input layer
        reduction (float): Reduction factor

    Returns:
        keras.layers.Layer: Output layer of the transition block
    """
    x = keras.layers.BatchNormalization()(x) # Batch normalization: normalize the activations of the previous layer at each batch
    x = keras.layers.Activation('relu')(x) # ReLU activation function
    x = keras.layers.Conv2D(int(x.shape[-1] * reduction), 1, use_bias=False)(x) # 1x1 convolutional layer
    x = keras.layers.AveragePooling2D(2, strides=2)(x) # Average pooling layer
    return x

def conv_block(x, growth_rate):
    """
    Create a convolutional block with the specified growth rate.
    A convolutional block is a block that applies a series of convolutional layers and concatenates the outputs.

    Args:
        x (keras.layers.Layer): Input layer
        growth_rate (int): Growth rate

    Returns:
        keras.layers.Layer: Output layer of the convolutional block
    """
    x1 = keras.layers.BatchNormalization()(x)
    x1 = keras.layers.Activation('relu')(x1)
    x1 = keras.layers.Conv2D(4 * growth_rate, 1, use_bias=False)(x1)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = keras.layers.Activation('relu')(x1)
    x1 = keras.layers.Conv2D(growth_rate, 3, padding='same', use_bias=False)(x1)
    x = keras.layers.Concatenate()([x, x1])
    return x

def DenseNet(blocks, growth_rate=32, reduction=0.5, input_shape=(64, 64, 1)):
    """
    Create a DenseNet model with the specified parameters.
    A DenseNet is a CNN that uses dense blocks and transition blocks instead of standard convolutional layers.

    Args:
        blocks (list): List of the number of blocks in each dense block
        growth_rate (int, optional): Growth rate. Defaults to 32.
        reduction (float, optional): Reduction factor. Defaults to 0.5.
        input_shape (tuple, optional): Shape of the input layer. Defaults to (64, 64, 1).

    Returns:
        keras.Model: DenseNet model
    """
    input_layer = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(2 * growth_rate, 7, strides=2, padding='same', use_bias=False)(input_layer)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    for block in blocks:
        x = dense_block(x, block, growth_rate)
        x = transition_block(x, reduction)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    output_layer = keras.layers.Dense(128, activation='relu')(x)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model