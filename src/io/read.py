"""
This file contains the functions to read from the database of random pulses, and format the data to be used in the neural network training.

The database is a csv file with the following structure:
    TBP, E, Tmn
where:
    TBP: Time between pulses
    E: Pulse in the time domain real part, then imaginary part
    Tmn: SHG-FROG trace of the pulse

E is a 2N-dimensional vector, and Tmn is a NxN matrix. N is given by the user.
"""

import tensorflow as tf
import numpy as np

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
    It returns a dataset with the input data (train) and the target data (target)

    Note that we also have the TBP of the pulses in the first column of the db.
    We want to save them in a separate array, so we can use them later.

    The function also normalizes each trace, by dividing by the maximum value.
    The real part and imaginary part of the pulse are normalized separately,
    dividing by the maximum value of each part.
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
