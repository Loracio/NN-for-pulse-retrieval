"""
In this test we will try to use the sweep feature of Weights & Biases to find the best
hyperparameters for our NN. 

We will use the same NN as in the simpleNNexample.py script, but we will try to find the best
number of hidden neurons and the best activation function for the hidden layer, as well as the
best optimizer, batch size and learning rate.
"""
import numpy as np
from readFromDB import load_data

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
    dense_layer = keras.layers.Dense(hidden_layer_neurons, activation=activation)(flatten_layer)
    for i in range(number_hidden_layers - 1):
        dense_layer = keras.layers.Dense(hidden_layer_neurons, activation=activation)(dense_layer)
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
def train(train_dataset, test_dataset, model, optimizer, loss_fn, train_acc_metric, test_acc_metric, epochs=10, log_step=200, val_log_step=200):
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait before stopping if no improvement in validation loss
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

# Sweep process
def sweep_train():

    # Initialize Weights & Biases
    run = wandb.init(config=sweep_config)

    # Specify the other hyperparameters to the configuration, if any
    wandb.config.log_step = LOG_STEP
    wandb.config.val_log_step = VAL_LOG_STEP
    # The architecture name is given by the number of hidden layers, neurons and activation function. Also the optimizer, learning rate and batch size
    # wandb.config.architecture_name = f"L={wandb.config.hidden_layer_number}_N={wandb.config.hidden_layer_neurons}_A={wandb.config.activation}_OP={wandb.config.optimizer}_LR={wandb.config.learning_rate}_BATCHSIZE={wandb.config.batch_size}"
    # wandb.config.dataset_name = f"{NUMBER_OF_PULSES}_randomPulses_N{N}"

    # Load dataset
    pulse_dataset = load_data(FILE_PATH, N, NUMBER_OF_PULSES)

    def select_yz(x, y, z):
        return (y, z)

    pulse_dataset = pulse_dataset.map(select_yz)

    # Split the dataset into train and test, shuffle and batch the train dataset
    train_dataset = pulse_dataset.take(int(0.75 * NUMBER_OF_PULSES)).shuffle(buffer_size=NUMBER_OF_PULSES).batch(wandb.config.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = pulse_dataset.skip(int(0.75 * NUMBER_OF_PULSES)).batch(wandb.config.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # initialize model
    model = make_model(number_hidden_layers=wandb.config.hidden_layer_number, 
                       hidden_layer_neurons=wandb.config.hidden_layer_neurons, 
                       activation=wandb.config.activation)

    train(train_dataset,
          val_dataset, 
          model,
          keras.optimizers.get(wandb.config.optimizer),
          keras.losses.get(wandb.config.loss),
          tf.keras.metrics.MeanSquaredError(), # In this case the test accuracy is the same as the test loss
          tf.keras.metrics.MeanSquaredError(), # In this case the train accuracy is the same as the test loss
          epochs=wandb.config.epochs, 
          log_step=wandb.config.log_step, 
          val_log_step=wandb.config.val_log_step)

if __name__ == '__main__':

    N = 64
    NUMBER_OF_PULSES = 2500
    FILE_PATH = f"./src/db/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        },
        'parameters': {
            'epochs': {
                'values': [25]
            },
            'batch_size': {
                'values': [16, 32, 64, 128, 256]
            },
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            'hidden_layer_neurons': {
                'values': [int(3 * N * N / 4), int(4 * N * N / 5), int( N * N / 2)]
            },
            'learning_rate': {
                'values': [0.1, 0.01, 0.001, 0.0001]
            },
            'activation': {
                'values': ['sigmoid', 'tanh', 'relu']
            },
            'hidden_layer_number': {
                'values': [1, 2, 3, 4, 5]
            },
            'loss': {
                'values': ['mse']
            },
        }
    }

    

    # Config for wandb
    LOG_STEP = 50
    VAL_LOG_STEP = 50

    sweep_id = wandb.sweep(sweep_config, project="SweepTest_SimpleNN")
    wandb.agent(sweep_id, function=sweep_train, count=33)