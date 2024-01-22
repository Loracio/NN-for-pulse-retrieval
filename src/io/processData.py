import tensorflow as tf

def process_data(N, NUMBER_OF_PULSES, pulse_dataset, training_size, BATCH_SIZE, SHUFFLE_BUFFER_SIZE=None):
    """
    In this function the data is processed.
    Data is batched and shuffled, dividing it into train and test sets.

    Args:
        N (int): Number of time steps
        NUMBER_OF_PULSES (int): Number of pulses in the database
        pulse_dataset (tf.data.Dataset): Dataset containing the pulses

    Returns:
        train_dataset (tf.data.Dataset): Dataset containing the training pulses
        test_dataset (tf.data.Dataset): Dataset containing the test pulses
    """
    if SHUFFLE_BUFFER_SIZE is None:
        SHUFFLE_BUFFER_SIZE = NUMBER_OF_PULSES

    # Select the y and z data from pulse dataset, which contain the SHG-FROG trace and the electric field of the pulse in the time domain
    pulse_dataset = pulse_dataset.map(lambda x, y, z: (y, z))

    # Split the dataset into train and test, shuffle and batch the train dataset
    train_dataset = pulse_dataset.take(int(training_size * NUMBER_OF_PULSES)).shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = pulse_dataset.skip(int(training_size * NUMBER_OF_PULSES)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, test_dataset