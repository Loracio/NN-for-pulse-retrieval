"""
This file contains the functions to read from the database of random pulses, and format the data to be used in the neural network training.

The database is a csv file with the following structure:
    TBP, E, Tmn
where:
    TBP: Time bandwidth product of the pulse
    E: Pulse in the time domain real part, then imaginary part
    Tmn: SHG-FROG trace of the pulse

E is a 2N-dimensional vector, and Tmn is a NxN matrix. N is given by the user.
"""

import tensorflow as tf
import numpy as np

from ..utils import fourier_utils

def load_data(FILE_PATH, N, NUMBER_OF_PULSES):
    """
    This function preprocesses the data from the database, iterating over it.
    It returns a dataset with the input data (train) and the target data (target)

    Note that we also have the TBP of the pulses in the first column of the db.
    We want to save them in a separate array, so we can use them later.
    """
    # Create a record_defaults with 1 + 2N + N*N elements that are tf.float32
    db_record_defaults = [tf.float32] * (1 + 2*N + N*N)

    # Read the database
    pulse_db = tf.data.experimental.CsvDataset(
        FILE_PATH, record_defaults=db_record_defaults, header=False)

    # Create empty datasets
    tbp_dataset = tf.data.Dataset.from_tensor_slices([])
    train_dataset = tf.data.Dataset.from_tensor_slices([])
    target_dataset = tf.data.Dataset.from_tensor_slices([])

    # Iterate over the database
    for i, pulse in enumerate(pulse_db):
        # Save the TBP in the tbp_dataset
        tbp_dataset = tbp_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.reshape(pulse[0], (1,))))

        # Save the SHG-FROG trace in the train_dataset
        train_dataset = train_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.reshape(pulse[2*N + 1:], (1, N, N))))

        # Save the pulse in the target_dataset
        target_dataset = target_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.reshape(pulse[1:2*N + 1], (1, 2*N))))

    # Create the final dataset
    dataset = tf.data.Dataset.zip((tbp_dataset, train_dataset, target_dataset))

    return dataset

def load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES):
    """
    This function preprocesses the data from the database, iterating over it.
    It returns a tensorflow dataset.

    Note that we also have the TBP of the pulses in the first column of the db.
    We want to save them in a separate array, so we can use them later.

    The function also normalizes each trace, by dividing by the maximum value.
    The real part and imaginary part of the pulse are normalized by dividing by
    the maximum absolute value (module) of the complex number.

    Args:
        FILE_PATH: str
            Path to the database file
        N: int
            Number of points in the SHG-FROG trace
        NUMBER_OF_PULSES: int
            Number of pulses in the database

    Returns:
        dataset: tf.data.Dataset
            Dataset with the pulse database
    """
    # Create a record_defaults with 1 + 2N + N*N elements that are tf.float32
    db_record_defaults = [tf.float32] * (1 + 2*N + N*N)

    # Read the database
    pulse_db = tf.data.experimental.CsvDataset(
        FILE_PATH, record_defaults=db_record_defaults, header=False)

    # Create empty datasets
    tbp_dataset = tf.data.Dataset.from_tensor_slices([])
    train_dataset = tf.data.Dataset.from_tensor_slices([])
    target_dataset = tf.data.Dataset.from_tensor_slices([])

    # Iterate over the database
    for i, pulse in enumerate(pulse_db):
        # Save the TBP in the tbp_dataset
        tbp_dataset = tbp_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.reshape(pulse[0], (1,))))

        # Save the SHG-FROG trace in the train_dataset and normalize
        shg_frog_trace = tf.reshape(pulse[2*N + 1:], (1, N, N))
        normalized_trace = shg_frog_trace / tf.reduce_max(tf.abs(shg_frog_trace))
        train_dataset = train_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(normalized_trace))

        # Save the pulse in the target_dataset and normalize
        pulse_real = tf.reshape(pulse[1:N + 1], (1, N))
        pulse_imag = tf.reshape(pulse[N + 1:2*N + 1], (1, N))

        # Combine real and imaginary parts into complex numbers
        pulse_complex = tf.complex(pulse_real, pulse_imag)

        # Find the maximum absolute value (module) of the complex numbers
        max_module = tf.reduce_max(tf.abs(pulse_complex))

        # Normalize the real and imaginary parts by the maximum module
        normalize_pulse_real = pulse_real / max_module
        normalize_pulse_imag = pulse_imag / max_module

        normalized_pulse = tf.concat([normalize_pulse_real, normalize_pulse_imag], axis=1)
        target_dataset = target_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(normalized_pulse))

    # Create the final dataset
    dataset = tf.data.Dataset.zip((tbp_dataset, train_dataset, target_dataset))

    return dataset

def _parse_function(example_proto):
    # Define the features in the TFRecord file
    feature_description = {
        'tbp': tf.io.FixedLenFeature([], tf.string),
        'real_field': tf.io.FixedLenFeature([], tf.string),
        'imag_field': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input tf.Example proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode the tensors from their serialized form
    tbp = tf.io.parse_tensor(parsed_example['tbp'], out_type=tf.float64)
    real_field = tf.io.parse_tensor(parsed_example['real_field'], out_type=tf.float64)
    imag_field = tf.io.parse_tensor(parsed_example['imag_field'], out_type=tf.float64)
    
    return real_field, imag_field, tbp

def read_tfrecord(FILE_PATH, N, NUMBER_OF_PULSES, BATCH_SIZE, add_noise=False, noise_level=0.01, mask=False, mask_tolerance=1e-3):
    """
    Read the TFRecord file and return a dataset with the pulses and their SHG-FROG traces.
    The pulses come already normalized, but the traces have to be normalized.
    There is an option to add Gaussian noise to the traces.
    Traces are computed and normalized (after adding the noise if needed) in this function.


    Args:
        FILE_PATH: str
            Path to the TFRecord file
        N: int
            Number of points in the SHG-FROG trace
        NUMBER_OF_PULSES: int
            Number of pulses in the database
        BATCH_SIZE: int
            Size of the batches to use in the dataset
        norm_traces: str
            Option for normalizing the traces
        add_noise: bool
            Option for adding Gaussian noise to the traces
        noise_level: float
            Level of noise to add to the traces (percentage of the maximum value of the trace)
        mask: bool
            Option for applying a mask to the noise
        mask_tolerance: float
            Tolerance for the mask

    Returns:
        dataset: tf.data.Dataset
            Dataset with the pulse database
    """
    # Create a dataset from the TFRecord file
    raw_dataset = tf.data.TFRecordDataset(FILE_PATH)

    # Map the parse function over the dataset
    parsed_dataset = raw_dataset.map(_parse_function)

    # Create empty datasets
    tbp_dataset = tf.data.Dataset.from_tensor_slices([])
    field_dataset = tf.data.Dataset.from_tensor_slices([])

    for tbp, real_field, imag_field in parsed_dataset:

        # Save the TBP in the tbp_dataset
        tbp_dataset = tbp_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.reshape(tf.cast(tbp, tf.float32), (1,))))

        # Concat the real and imaginary parts into a single tensor
        pulse = tf.concat([tf.reshape(real_field, (1, N)), tf.reshape(imag_field, (1,N))], axis=1)
        
        # Add the pulse to the field_dataset
        field_dataset = field_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.cast(pulse, tf.float32)))

    # Compute trace of the electric fields
    fourier = fourier_utils(N, 1/N)
    # We pass it as a batch for performance (it's faster to compute the fourier transform of a batch of pulses)
    # and less memory intensive
    trace_dataset = field_dataset.batch(BATCH_SIZE).map(fourier.compute_trace)

    # Add noise to the traces if needed
    if add_noise:
        trace_dataset = trace_dataset.map(lambda x: fourier.add_gaussian_noise(x, noise_level, mask=mask, mask_tolerance=mask_tolerance))

    # Norm the traces by dividing by the maximum value of each trace
    trace_dataset = trace_dataset.map(lambda x: x / tf.reduce_max(tf.abs(x), axis=[1, 2], keepdims=True))

    # Unbatch the dataset
    trace_dataset = trace_dataset.unbatch()

    dataset = tf.data.Dataset.zip((tbp_dataset, trace_dataset, field_dataset))

    return dataset

def _parse_functionNoisy(example_proto):
    # Define the features in the TFRecord file
    feature_description = {
        'real_field': tf.io.FixedLenFeature([], tf.string),
        'imag_field': tf.io.FixedLenFeature([], tf.string),
        'noisy_trace': tf.io.FixedLenFeature([], tf.string),
        'real_retrieved_field': tf.io.FixedLenFeature([], tf.string),
        'imag_retrieved_field': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input tf.Example proto using the dictionary above.
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode the tensors from their serialized form
    real_field = tf.io.parse_tensor(parsed_example['real_field'], out_type=tf.float64)
    imag_field = tf.io.parse_tensor(parsed_example['imag_field'], out_type=tf.float64)
    noisy_trace = tf.io.parse_tensor(parsed_example['noisy_trace'], out_type=tf.float64)
    real_retrieved_field = tf.io.parse_tensor(parsed_example['real_retrieved_field'], out_type=tf.float64)
    imag_retrieved_field = tf.io.parse_tensor(parsed_example['imag_retrieved_field'], out_type=tf.float64)

    return real_field, imag_field, noisy_trace, real_retrieved_field, imag_retrieved_field

def read_tfrecord_noisyTraces(FILE_PATH, N, NUMBER_OF_PULSES, BATCH_SIZE):
    """
    Read the TFRecord file and return a dataset with the pulses and their SHG-FROG noisy traces.
    The pulses come already normalized.
    Traces are computed and normalized in this function to compare to the noisy traces.


    Args:
        FILE_PATH: str
            Path to the TFRecord file
        N: int
            Number of points in the SHG-FROG trace
        NUMBER_OF_PULSES: int
            Number of pulses in the database
        BATCH_SIZE: int
            Size of the batches to use in the dataset

    Returns:
        dataset: tf.data.Dataset
            Dataset with the pulse database
    """
    # Create a dataset from the TFRecord file
    raw_dataset = tf.data.TFRecordDataset(FILE_PATH)

    # Map the parse function over the dataset
    parsed_dataset = raw_dataset.map(_parse_functionNoisy)

    # Create empty datasets
    field_dataset = tf.data.Dataset.from_tensor_slices([])
    retrieved_field_dataset = tf.data.Dataset.from_tensor_slices([])

    # noisy_trace_dataset = tf.data.Dataset.from_tensor_slices([])
    noisy_trace_dataset = parsed_dataset.map(lambda real_field, imag_field, noisy_trace, real_retrieved_field, imag_retrieved_field: noisy_trace)

    for real_field, imag_field, noisy_trace, real_retrieved_field, imag_retrieved_field in parsed_dataset:
        # Concat the real and imaginary parts into a single tensor
        pulse = tf.concat([tf.reshape(real_field, (1, N)), tf.reshape(imag_field, (1,N))], axis=1)

        # Add the pulse to the field_dataset
        field_dataset = field_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.cast(pulse, tf.float32)))

        # Concat the real and imaginary parts into a single tensor
        retrieved_field = tf.concat([tf.reshape(real_retrieved_field, (1, N)), tf.reshape(imag_retrieved_field, (1,N))], axis=1)

        # Add the retrieved field to the retrieved_field_dataset
        retrieved_field_dataset = retrieved_field_dataset.concatenate(
            tf.data.Dataset.from_tensor_slices(tf.cast(retrieved_field, tf.float32)))
        

    # Compute trace of the electric fields
    fourier = fourier_utils(N, 1/N)
    # We pass it as a batch for performance (it's faster to compute the fourier transform of a batch of pulses)
    # and less memory intensive
    trace_dataset = field_dataset.batch(BATCH_SIZE).map(fourier.compute_trace)

    # Norm the traces by dividing by the maximum value of each trace
    trace_dataset = trace_dataset.map(lambda x: x / tf.reduce_max(tf.abs(x), axis=[1, 2], keepdims=True))

    # Unbatch the dataset
    trace_dataset = trace_dataset.unbatch()

    dataset = tf.data.Dataset.zip((noisy_trace_dataset, trace_dataset, field_dataset, retrieved_field_dataset))

    return dataset