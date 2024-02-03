import tensorflow as tf
import numpy as np
import wandb
from .train_step import train_step_MLP, train_step_MLP_custom_loss, train_step_CNN_custom_loss
from .test_step import test_step_MLP, test_step_MLP_custom_loss, test_step_CNN_custom_loss

def train_MLP(train_dataset, test_dataset, model, optimizer, loss_fn, train_acc_metric, test_acc_metric, epochs, log_step, val_log_step, patience):
    """
    Trainin step for a MLP model. Updates the weights of the model using the gradients computed by the loss function.
    Saves the training and validation loss and accuracy in wandb.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        model (tf.keras.Model): Model to train.
        optimizer (tf.keras.optimizers): Optimizer to use.
        loss_fn (tf.keras.losses): Loss function to use.
        train_acc_metric (tf.keras.metrics): Training accuracy metric.
        test_acc_metric (tf.keras.metrics): Test accuracy metric.
        epochs (int): Number of epochs to train.
        log_step (int): Number of steps to log training metrics.
        val_log_step (int): Number of steps to log validation metrics.
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step_MLP(x_batch_train, y_batch_train,
                                    model, optimizer,
                                    loss_fn, train_acc_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step_MLP(x_batch_val, y_batch_val,
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
            print("Early stopping due to no improvement in validation loss")
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
                   'train_loss': np.mean(train_loss),
                   'train_acc': float(train_acc),
                   'test_loss': np.mean(val_loss),
                   'test_acc': float(test_acc)
                   })

def train_MLP_custom_loss(train_dataset, test_dataset, model, optimizer, custom_loss_fn, train_acc_metric, test_acc_metric, epochs, log_step, val_log_step, patience):
    """
    Trainin step for a MLP model. Updates the weights of the model using the gradients computed by the custom loss function (trace mse).
    Saves the training and validation loss and accuracy in wandb.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        model (tf.keras.Model): Model to train.
        optimizer (tf.keras.optimizers): Optimizer to use.
        custom_loss_fn (tf.keras.losses): Loss function to use.
        train_acc_metric (tf.keras.metrics): Training accuracy metric.
        test_acc_metric (tf.keras.metrics): Test accuracy metric.
        epochs (int): Number of epochs to train.
        log_step (int): Number of steps to log training metrics.
        val_log_step (int): Number of steps to log validation metrics.
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step_MLP_custom_loss(x_batch_train, y_batch_train,
                                    model, optimizer,
                                    custom_loss_fn, train_acc_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step_MLP_custom_loss(x_batch_val, y_batch_val,
                                       model, custom_loss_fn,
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
            print("Early stopping due to no improvement in validation loss")
            break

        # Display loss at the end of each epoch (scientific notation with 4 decimal places)
        print("Training loss over epoch: %.4e" % (np.mean(train_loss),))
        print("Validation loss: %.4e" % (avg_val_loss,))

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
                   'train_loss': np.mean(train_loss),
                   'train_acc': float(train_acc),
                   'test_loss': np.mean(val_loss),
                   'test_acc': float(test_acc)
                   })

def train_CNN_custom_loss(train_dataset, test_dataset, model, optimizer, custom_loss_fn, train_acc_metric, test_acc_metric, epochs, log_step, val_log_step, patience):
    """
    Trainin step for a CNN model. Updates the weights of the model using the gradients computed by the custom loss function (trace mse).
    Saves the training and validation loss and accuracy in wandb.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        model (tf.keras.Model): Model to train.
        optimizer (tf.keras.optimizers): Optimizer to use.
        custom_loss_fn (tf.keras.losses): Loss function to use.
        train_acc_metric (tf.keras.metrics): Training accuracy metric.
        test_acc_metric (tf.keras.metrics): Test accuracy metric.
        epochs (int): Number of epochs to train.
        log_step (int): Number of steps to log training metrics.
        val_log_step (int): Number of steps to log validation metrics.
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step_CNN_custom_loss(x_batch_train, y_batch_train,
                                    model, optimizer,
                                    custom_loss_fn, train_acc_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step_CNN_custom_loss(x_batch_val, y_batch_val,
                                       model, custom_loss_fn,
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
            print("Early stopping due to no improvement in validation loss")
            break

        # Display loss at the end of each epoch (scientific notation with 4 decimal places)
        print("Training loss over epoch: %.4e" % (np.mean(train_loss),))
        print("Validation loss: %.4e" % (avg_val_loss,))

        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4e" % (float(train_acc),))

        test_acc = test_acc_metric.result()
        print("Validation acc: %.4e" % (float(test_acc),))

        # Reset metrics at the end of each epoch
        train_acc_metric.reset_states()
        test_acc_metric.reset_states()


        # log metrics using wandb.log
        wandb.log({'epochs': epoch,
                   'train_loss': np.mean(train_loss),
                   'train_acc': float(train_acc),
                   'test_loss': np.mean(val_loss),
                   'test_acc': float(test_acc)
                   })
