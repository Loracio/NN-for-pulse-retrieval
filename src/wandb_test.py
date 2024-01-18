"""
In this example, we will use Weights & Biases to log the training process of the NN.

The Neural Network will try to predict the Electric field of a pulse in the time domain from its SHG-FROG trace.
The input data is the SHG-FROG trace of the pulse (NxN vector), and the target data is the
Electric field of the pulse in the time domain (2N-dimensional vector containing
the real and imaginary parts of the pulse).
"""
import numpy as np
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
run = wandb.init(project="simpleNNexample", config=config, name='Three dense layers with tanh')
config = wandb.config

# Load pulse database
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

# Define the train step
def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        results = model(x, training=True)
        loss_value = loss_fn(y, results)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, results)

    return loss_value

# Define the test step
def test_step(x, y, model, loss_fn, test_acc_metric):
    val_results = model(x, training=False)
    loss_value = loss_fn(y, val_results)

    test_acc_metric.update_state(y, val_results)

    return loss_value

# Define the training loop
def train(train_dataset, test_dataset, model, optimizer, train_acc_metric, test_acc_metric, epochs=EPOCHS, log_step=LOG_STEP, val_log_step=VAL_LOG_STEP):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       test_acc_metric)
            average_loss_value = tf.reduce_mean(val_loss_value)
            val_loss.append(float(average_loss_value))
            
        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        test_acc = test_acc_metric.result()
        print("Validation acc: %.4f" % (float(test_acc),))

        # Reset metrics at the end of each epoch
        train_acc_metric.reset_states()
        test_acc_metric.reset_states()

        # log metrics using wandb.log
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'test_loss': np.mean(val_loss),
                   'test_acc':float(test_acc)})

# Initialize model.
model = make_model()

# Instantiate an optimizer to train the model.
# We take it from the config dictionary
optimizer = keras.optimizers.get(config.optimizer)
# Instantiate a loss function, taken from the config dictionary
loss_fn = keras.losses.get(config.loss)

# Prepare the metrics. Taken from the config dictionary
train_acc_metric = tf.keras.metrics.RootMeanSquaredError()
test_acc_metric = tf.keras.metrics.RootMeanSquaredError()

train(train_dataset,
      test_dataset, 
      model,
      optimizer,
      train_acc_metric,
      test_acc_metric,
      epochs=config.epochs, 
      log_step=config.log_step, 
      val_log_step=config.val_log_step)

run.finish()  # Finish wandb run

# Save the model
model.save('./src/models/simpleNNexample.h5')