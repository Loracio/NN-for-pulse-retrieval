import numpy as np
from readAndNormFromDB import load_data

import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

# Build the model


def make_model(number_hidden_layers, hidden_layer_neurons, activation):
    # Hidden layer neurons and activation are given by the sweep
    inputs = keras.Input(shape=(N, N), name="input")
    flatten_layer = keras.layers.Flatten()(inputs)
    # Add the hidden layers given by the sweep
    dense_layer = keras.layers.Dense(
        hidden_layer_neurons, activation=activation)(flatten_layer)
    for i in range(number_hidden_layers - 1):
        dense_layer = keras.layers.Dense(
            hidden_layer_neurons, activation=activation)(dense_layer)
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


def train(train_dataset, test_dataset, model, optimizer, loss_fn, train_acc_metric, test_acc_metric, epochs=10, log_step=200, val_log_step=200, early_stopping=10):
    best_val_loss = float('inf')
    # Number of epochs to wait before stopping if no improvement in validation loss
    patience = early_stopping
    patience_counter = 0

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

        avg_val_loss = np.mean(val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print("Early stopping due to no improvement in validation loss\n")
            break

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
                   'test_loss': np.mean(val_loss)
                   })


if __name__ == '__main__':

    N = 64
    NUMBER_OF_PULSES = 1000
    FILE_PATH = f"./src/db/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"

    config = {

        'epochs': 50,
        'batch_size': 8,
        'optimizer': 'sgd',
        'hidden_layer_neurons': 3072,
        'learning_rate': 0.001,
        'activation': 'relu',
        'hidden_layer_number': 3,
        'loss': 'mse',
        'early_stopping': 10,
        'log_step': 50,
        'val_log_step': 50
    }

    # Initialize Weights & Biases
    run = wandb.init(project="simpleNNexample",
                     config=config, name='modBest MLP of sweep, normData')

    # Load dataset
    pulse_dataset = load_data(FILE_PATH, N, NUMBER_OF_PULSES)

    def select_yz(x, y, z):
        return (y, z)

    pulse_dataset = pulse_dataset.map(select_yz)

    # Split the dataset into train and test, shuffle and batch the train dataset
    train_dataset = pulse_dataset.take(int(0.75 * NUMBER_OF_PULSES)).shuffle(
        buffer_size=NUMBER_OF_PULSES).batch(wandb.config.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = pulse_dataset.skip(int(0.75 * NUMBER_OF_PULSES)).batch(
        wandb.config.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # initialize model
    model = make_model(number_hidden_layers=wandb.config.hidden_layer_number,
                       hidden_layer_neurons=wandb.config.hidden_layer_neurons,
                       activation=wandb.config.activation)

    train(train_dataset,
          val_dataset,
          model,
          keras.optimizers.get(wandb.config.optimizer),
          keras.losses.get(wandb.config.loss),
          # In this case the test accuracy is the same as the test loss
          tf.keras.metrics.MeanSquaredError(),
          # In this case the train accuracy is the same as the test loss
          tf.keras.metrics.MeanSquaredError(),
          early_stopping=wandb.config.early_stopping,
          epochs=wandb.config.epochs,
          log_step=wandb.config.log_step,
          val_log_step=wandb.config.val_log_step)

    run.finish()  # Finish wandb run
