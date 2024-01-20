"""
In this example, we will use Weights & Biases to log the training process of the NN.

The Neural Network will try to predict the Electric field of a pulse in the time domain from its SHG-FROG trace.
The input data is the SHG-FROG trace of the pulse (NxN vector), and the target data is the
Electric field of the pulse in the time domain (2N-dimensional vector containing
the real and imaginary parts of the pulse).
"""
import numpy as np
from readAndNormFromDB import load_data
from utils import compute_trace_error

import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

# Config for wandb
EPOCHS = 50
LOG_STEP = 50  # Log metrics every LOG_STEP batches
VAL_LOG_STEP = 50  # Log validation metrics every VAL_LOG_STEP batches
BATCH_SIZE = 16  # Batch size
OPTIMIZER = 'sgd'
LOSS = 'mse'
METRICS = 'mse'

config = {
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'log_step': LOG_STEP,
    'val_log_step': VAL_LOG_STEP,
    'optimizer': OPTIMIZER,
    'loss': LOSS,
    'metrics': [METRICS, 'trace_error']
}

# Initialize Weights & Biases
run = wandb.init(project="simpleNNexample", config=config,
                 name='1xConv2D(8,8) + 3DescentFCL + sgd, Norm Data')
config = wandb.config

# Load pulse database
N = 64
NUMBER_OF_PULSES = 2500
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
    # input_shape = (N, N, 1) for Conv2D
    inputs = keras.Input(shape=(N, N, 1), name="input")
    # Lets make a convolutional neural network with 3 hidden layers
    conv2d_layer = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    # maxpool2d_layer = keras.layers.MaxPooling2D((2, 2))(conv2d_layer)
    # conv2d_layer = keras.layers.Conv2D(
        # 16, (3, 3), activation='relu')(maxpool2d_layer)
    # maxpool2d_layer = keras.layers.MaxPooling2D((2, 2))(conv2d_layer)
    conv2d_layer = keras.layers.Conv2D(
        8, (3, 3), activation='relu')(conv2d_layer)
    # maxpool2d_layer = keras.layers.MaxPooling2D((2, 2))(conv2d_layer)
    flatten_layer = keras.layers.Flatten()(conv2d_layer)
    dense_layer = keras.layers.Dense(3072, activation='relu')(flatten_layer)
    dense_layer = keras.layers.Dense(1536, activation='relu')(dense_layer)
    dense_layer = keras.layers.Dense(768, activation='relu')(dense_layer)
    outputs = keras.layers.Dense(2 * N, name="output")(dense_layer)

    model = keras.Model(inputs=inputs, outputs=outputs)
    # Print model summary
    model.summary()

    return model

# Define the train step


def train_step(x, y, model, optimizer, loss_fn, train_acc_metric, trace_error_metric):
    with tf.GradientTape() as tape:
        results = model(x, training=True)
        loss_value = loss_fn(y, results)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, results)

    # trace_error = compute_trace_error(x, results)
    # print(f"Trace error: {trace_error}\n" )
    # trace_error_metric.update_state(trace_error)

    return loss_value

# Define the test step


def test_step(x, y, model, loss_fn, test_acc_metric, trace_error_metric):
    val_results = model(x, training=False)
    loss_value = loss_fn(y, val_results)
    # trace_error = compute_trace_error(x, val_results)

    # trace_error_metric.update_state(trace_error)

    # test_acc_metric.update_state(y, val_results)

    return loss_value

# Define the training loop


def train(train_dataset, test_dataset, model, optimizer, train_acc_metric, test_acc_metric, train_trace_error_metric, test_trace_error_metric, epochs=EPOCHS, log_step=LOG_STEP, val_log_step=VAL_LOG_STEP):
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait before stopping
    patience_counter = 0

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train,
                                    model, optimizer,
                                    loss_fn, train_acc_metric, train_trace_error_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val,
                                       model, loss_fn,
                                       test_acc_metric, test_trace_error_metric)
            average_loss_value = tf.reduce_mean(val_loss_value)
            val_loss.append(float(average_loss_value))

        avg_val_loss = np.mean(val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print("Early stopping due to no improvement in validation loss")
            break

        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        test_acc = test_acc_metric.result()
        print("Validation acc: %.4f" % (float(test_acc),))

        # train_trace_error = train_trace_error_metric.result()
        # print("Training trace error over epoch: %.4f" %
        #       (float(train_trace_error),))

        # test_trace_error = test_trace_error_metric.result()
        # print("Validation trace error: %.4f" % (float(test_trace_error),))

        # Reset metrics at the end of each epoch
        train_acc_metric.reset_states()
        test_acc_metric.reset_states()

        # train_trace_error_metric.reset_states()
        # test_trace_error_metric.reset_states()

        # log metrics using wandb.log
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc),
                   'test_loss': np.mean(val_loss),
                   'test_acc': float(test_acc),
                #    'train_trace_error': float(train_trace_error),
                #    'test_trace_error': float(test_trace_error)
                   })


# Initialize model.
model = make_model()

# Instantiate an optimizer to train the model.
# We take it from the config dictionary
optimizer = keras.optimizers.get(config.optimizer)
# Instantiate a loss function, taken from the config dictionary
loss_fn = keras.losses.get(config.loss)

# Prepare the metrics. Taken from the config dictionary
train_acc_metric = tf.keras.metrics.MeanSquaredError()
test_acc_metric = tf.keras.metrics.MeanSquaredError()

train_trace_error_metric = tf.keras.metrics.Mean()
test_trace_error_metric = tf.keras.metrics.Mean()

train(train_dataset,
      test_dataset,
      model,
      optimizer,
      train_acc_metric,
      test_acc_metric,
      train_trace_error_metric,
      test_trace_error_metric,
      epochs=config.epochs,
      log_step=config.log_step,
      val_log_step=config.val_log_step)

run.finish()  # Finish wandb run

# Save the modelst
model.save('./src/models/simpleNNexample.h5')
