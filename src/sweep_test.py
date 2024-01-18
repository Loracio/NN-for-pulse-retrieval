"""
In this test we will try to use the sweep feature of Weights & Biases to find the best
hyperparameters for our NN. 

We will use the same NN as in the simpleNNexample.py script, but we will try to find the best
number of hidden neurons and the best activation function for the hidden layer, as well as the
best optimizer, batch size and learning rate.
"""
import numpy as np
import tqdm # For progress bar
from readFromDB import load_data

import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

# Config for wandb
EPOCHS = 10
LOG_STEP = 200
VAL_LOG_STEP = 50
BATCH_SIZE = 32
OPTIMIZER = 'adam'
LOSS = 'mse'
METRICS = 'mse'

config = {
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'log_step': LOG_STEP,
    'val_log_step': VAL_LOG_STEP,
    'optimizer': OPTIMIZER,
    'loss': LOSS,
    'metrics': METRICS
}

# Initialize Weights & Biases
run = wandb.init(project="SweepTest_SimpleNN", config=config, name='1st try')
config = wandb.config


# Load dataset
N = 64
NUMBER_OF_PULSES = 1000
FILE_PATH = f"./src/db/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"
pulse_dataset = load_data(FILE_PATH, N, NUMBER_OF_PULSES)

def select_yz(x, y, z):
    return (y, z)

pulse_dataset = pulse_dataset.map(select_yz)

# Split the dataset into train and test
train_dataset = pulse_dataset.take(int(0.8 * NUMBER_OF_PULSES))
test_dataset = pulse_dataset.skip(int(0.8 * NUMBER_OF_PULSES))

# Shuffle the train dataset
train_dataset = train_dataset.shuffle(buffer_size=NUMBER_OF_PULSES)

# Batch the datasets
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Build the model
def make_model():
    # Lets make it three dense layer deep with 2*N neurons each with tanh
    inputs = keras.Input(shape=(N, N), name="input")
    flatten_layer = keras.layers.Flatten()(inputs)
    dense_layer = keras.layers.Dense(2 * N, activation='tanh')(flatten_layer)
    dense_layer = keras.layers.Dense(2 * N, activation='tanh')(dense_layer)
    dense_layer = keras.layers.Dense(2 * N, activation='tanh')(dense_layer)
    outputs = keras.layers.Dense(2 * N, name="output")(dense_layer)

    return keras.Model(inputs=inputs, outputs=outputs)