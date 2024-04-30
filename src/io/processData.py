import tensorflow as tf
from .read import read_tfrecord, read_tfrecord_noisyTraces

def process_data(N, NUMBER_OF_PULSES, pulse_dataset, training_size, BATCH_SIZE, SHUFFLE_BUFFER_SIZE=None):
    """
    In this function the data is processed.
    Data is batched and shuffled, dividing it into train and test sets.

    Args:
        N (int): Number of samples
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
    train_dataset = pulse_dataset.take(int(training_size * NUMBER_OF_PULSES))
    test_dataset = pulse_dataset.skip(int(training_size * NUMBER_OF_PULSES))

    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, test_dataset

def process_data_tfrecord(N, NUMBER_OF_PULSES, FILE_PATH, TRAINING_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE=None, add_noise=False, noise_level=0.01, mask=False, mask_tolerance=1e-3):
    """
    Read the TFRecord file and process the data.

    Args:
        N (int): Number of samples
        NUMBER_OF_PULSES (int): Number of pulses in the database
        FILE_PATH (str): Path to the TFRecord file
        TRAINING_SIZE (float): Percentage of the dataset to use for training
        BATCH_SIZE (int): Size of the batches to use in the dataset
        norm_traces (str): Option for normalizing the traces
        SHUFFLE_BUFFER_SIZE (int): Size of the buffer to use for shuffling the dataset
        add_noise (bool): Option for adding Gaussian noise to the traces
        noise_level (float): Level of noise to add to the traces (percentage of the maximum value of the trace)
        mask (bool): Option for applying a mask to the noise
        mask_tolerance (float): Tolerance for the mask

    Returns:
        train_dataset (tf.data.Dataset): Dataset containing the training pulses
        test_dataset (tf.data.Dataset): Dataset containing the test pulses
    """
    pulse_dataset = read_tfrecord(FILE_PATH, N, NUMBER_OF_PULSES, BATCH_SIZE, add_noise, noise_level, mask=mask, mask_tolerance=mask_tolerance)

    return process_data(N, NUMBER_OF_PULSES, pulse_dataset, TRAINING_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)

def load_data_from_batches(FILE_PATH, N, NUMBER_OF_PULSES, SHUFFLE_BUFFER_SIZE=None):
    """
    Load the data from the TFRecord file that contains one batch of data.

    Args:
        FILE_PATH (str): Path to the TFRecord file
        N (int): Number of samples
        NUMBER_OF_PULSES (int): Number of pulses in the dataset
        SHUFFLE_BUFFER_SIZE (int): Size of the buffer to use for shuffling the dataset

    Returns:
        x_batch (tf.Tensor): Tensor containing the SHG-FROG traces
        y_batch (tf.Tensor): Tensor containing the pulses
    """
    pulse_dataset = read_tfrecord(FILE_PATH, N, NUMBER_OF_PULSES, NUMBER_OF_PULSES)

    if SHUFFLE_BUFFER_SIZE is None:
        SHUFFLE_BUFFER_SIZE = NUMBER_OF_PULSES

    # Select the y and z data from pulse dataset, which contain the SHG-FROG trace and the electric field of the pulse in the time domain
    x_batch = pulse_dataset.map(lambda x, y, z: y)
    y_batch = pulse_dataset.map(lambda x, y, z: z)

    # Convert from dataset to tensor
    x_batch = tf.convert_to_tensor(list(x_batch.as_numpy_iterator()))
    y_batch = tf.convert_to_tensor(list(y_batch.as_numpy_iterator()))

    return x_batch, y_batch


def process_data_tfrecord_noisyTraces(N, NUMBER_OF_PULSES, FILE_PATH, TRAINING_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE=None):
    """
    In this function the data is processed.
    Data is batched and shuffled, dividing it into train and test sets.

    Args:
        N (int): Number of samples
        NUMBER_OF_PULSES (int): Number of pulses in the database
        training_size (float): Percentage of the dataset to use for training
        BATCH_SIZE (int): Size of the batches to use in the dataset

    Returns:
        train_dataset (tf.data.Dataset): Dataset containing the training pulses and training noisy traces
        test_dataset (tf.data.Dataset): Dataset containing the test pulses and test noisy traces
        train_original_trace_dataset (tf.data.Dataset): Dataset containing the training original traces
        test_original_trace_dataset (tf.data.Dataset): Dataset containing the test original traces
    """

    pulse_dataset = read_tfrecord_noisyTraces(FILE_PATH, N, NUMBER_OF_PULSES, BATCH_SIZE)

    if SHUFFLE_BUFFER_SIZE is None:
        SHUFFLE_BUFFER_SIZE = NUMBER_OF_PULSES

    # The y and z data from pulse dataset, which contain the SHG-FROG trace and the electric field of the pulse in the time domain
    # The xdata contain the noisy traces
    original_trace_dataset = pulse_dataset.map(lambda x, y, z, t: y)
    retrieved_field_dataset = pulse_dataset.map(lambda x, y, z, t: t)
    pulse_dataset = pulse_dataset.map(lambda x, y, z, t: (x, z))
    

    # Split the dataset into train and test, shuffle and batch the train dataset
    train_dataset = pulse_dataset.take(int(TRAINING_SIZE * NUMBER_OF_PULSES))
    test_dataset = pulse_dataset.skip(int(TRAINING_SIZE * NUMBER_OF_PULSES))

    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    train_original_trace_dataset = original_trace_dataset.take(int(TRAINING_SIZE * NUMBER_OF_PULSES)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_original_trace_dataset = original_trace_dataset.skip(int(TRAINING_SIZE * NUMBER_OF_PULSES)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    train_retrieved_field_dataset = retrieved_field_dataset.take(int(TRAINING_SIZE * NUMBER_OF_PULSES)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_retrieved_field_dataset = retrieved_field_dataset.skip(int(TRAINING_SIZE * NUMBER_OF_PULSES)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, test_dataset, train_original_trace_dataset, test_original_trace_dataset, train_retrieved_field_dataset, test_retrieved_field_dataset