import tensorflow as tf
import numpy as np
import wandb
from .train_step import *
from .test_step import *

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

def train_MLP_intensity_loss(train_dataset, test_dataset, model, optimizer, intensity_loss_fn, train_acc_metric, test_acc_metric, epochs, log_step, val_log_step, patience):
    """
    Trainin step for a MLP model. Updates the weights of the model using the gradients computed by the intensity loss function (trace mse).
    Saves the training and validation loss and accuracy in wandb.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        model (tf.keras.Model): Model to train.
        optimizer (tf.keras.optimizers): Optimizer to use.
        intensity_loss_fn (tf.keras.losses): Loss function to use.
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
            loss_value = train_step_MLP_intensity_loss(x_batch_train, y_batch_train,
                                    model, optimizer,
                                    intensity_loss_fn, train_acc_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step_MLP_intensity_loss(x_batch_val, y_batch_val,
                                       model, intensity_loss_fn,
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
                   'Intensity train loss': np.mean(train_loss),
                   'MSE field train acc': float(train_acc),
                   'Intensity test loss': np.mean(val_loss),
                   'MSE field test acc': float(test_acc)
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


def train_joint_loss(train_dataset, test_dataset, model, optimizer, weight_trace_loss, trace_loss, weight_field_loss, field_loss, train_trace_metric, train_field_metric, train_intensity_metric, test_trace_metric, test_field_metric, test_intensity_metric, epochs, log_step, val_log_step, patience):
    """
    Training step that uses a joint custom loss that takes into account the trace MSE and the electric field MSE.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        model (tf.keras.Model): Model to train.
        optimizer (tf.keras.optimizers): Optimizer to use.
        weight_trace_loss (float): Weight for the trace loss.
        trace_loss (tf.keras.losses): Trace loss function to use.
        weight_field_loss (float): Weight for the field loss.
        field_loss (tf.keras.losses): Field loss function to use.
        trace_acc_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the trace.
        field_acc_metric (tf.keras.metrics): Accuracy metric that computes the MSe of the field.
        intensity_acc_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the intensity.
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
            loss_value = train_step_joint_loss(x_batch_train, y_batch_train,
                                    model, optimizer,
                                    weight_trace_loss, trace_loss, weight_field_loss, field_loss, train_trace_metric, train_field_metric, train_intensity_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step_joint_loss(x_batch_val, y_batch_val,
                                       model, weight_trace_loss, trace_loss, weight_field_loss, field_loss, test_trace_metric, test_field_metric, test_intensity_metric)
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
        train_trace_acc = train_trace_metric.result()
        train_field_acc = train_field_metric.result()
        train_intensity_acc = train_intensity_metric.result()
        print("Training trace acc over epoch: %.4e" % (float(train_trace_acc),))
        print("Training field acc over epoch: %.4e" % (float(train_field_acc),))
        print("Training intensity acc over epoch: %.4e" % (float(train_intensity_acc),))

        test_trace_acc = test_trace_metric.result()
        test_field_acc = test_field_metric.result()
        test_intensity_acc = test_intensity_metric.result()
        print("Validation trace acc: %.4e" % (float(test_trace_acc),))
        print("Validation field acc: %.4e" % (float(test_field_acc),))
        print("Validation intensity acc: %.4e" % (float(test_intensity_acc),))

        # Reset metrics at the end of each epoch
        train_trace_metric.reset_states()
        train_field_metric.reset_states()
        train_intensity_metric.reset_states()
        test_trace_metric.reset_states()
        test_field_metric.reset_states()
        test_intensity_metric.reset_states()


        # log metrics using wandb.log
        wandb.log({'Epochs': epoch,
                   'Train joint loss (trace MSE + field MSE)': np.mean(train_loss),
                   'Train trace MSE': float(train_trace_acc),
                   'Train field MSE': float(train_field_acc),
                   'Train intensity MSE': float(train_intensity_acc),
                   'Test joint loss (trace MSE + field MSE)': np.mean(val_loss),
                   'Test trace MSE': float(test_trace_acc),
                   'Test field MSE': float(test_field_acc),
                   'Test intensity MSE': float(test_intensity_acc)
                   })


def train_joint_loss_intensity(train_dataset, test_dataset, model, optimizer, weight_trace_loss, trace_loss, weight_intensity_loss, intensity_loss, train_trace_metric, train_field_metric, test_trace_metric, test_field_metric, train_intensity_metric, test_intensity_metric, epochs, log_step, val_log_step, patience):
    """
    Training step that uses a joint custom loss that takes into account the trace MSE and the electric field MSE.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        model (tf.keras.Model): Model to train.
        optimizer (tf.keras.optimizers): Optimizer to use.
        weight_trace_loss (float): Weight for the trace loss.
        trace_loss (tf.keras.losses): Trace loss function to use.
        weight_intensity_loss (float): Weight for the intensity loss.
        intensity_loss (tf.keras.losses): Intensity loss function to use.
        train_trace_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the trace.
        train_field_metric (tf.keras.metrics): Accuracy metric that computes the MSe of the field.
        train_intensity_metric (tf.keras.metrics): Accuracy metric that computes the MSe of the intensity.
        test_trace_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the trace.
        test_field_metric (tf.keras.metrics): Accuracy metric that computes the MSe of the field.
        test_intensity_metric (tf.keras.metrics): Accuracy metric that computes the MSe of the intensity.
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
            loss_value = train_step_joint_loss_intensity(x_batch_train, y_batch_train,
                                    model, optimizer,
                                    weight_trace_loss, trace_loss, weight_intensity_loss, intensity_loss, train_trace_metric, train_field_metric, train_intensity_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step_joint_loss_intensity(x_batch_val, y_batch_val,
                                       model, weight_trace_loss, trace_loss, weight_intensity_loss, intensity_loss, test_trace_metric, test_field_metric, test_intensity_metric)
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
        train_trace_acc = train_trace_metric.result()
        train_field_acc = train_field_metric.result()
        train_intensity_acc = train_intensity_metric.result()
        print("Training trace acc over epoch: %.4e" % (float(train_trace_acc),))
        print("Training field acc over epoch: %.4e" % (float(train_field_acc),))
        print("Training intensity acc over epoch: %.4e" % (float(train_intensity_acc),))

        test_trace_acc = test_trace_metric.result()
        test_field_acc = test_field_metric.result()
        test_intensity_acc = test_intensity_metric.result()
        print("Validation trace acc: %.4e" % (float(test_trace_acc),))
        print("Validation field acc: %.4e" % (float(test_field_acc),))
        print("Validation intensity acc: %.4e" % (float(test_intensity_acc),))

        # Reset metrics at the end of each epoch
        train_trace_metric.reset_states()
        train_field_metric.reset_states()
        train_intensity_metric.reset_states()
        test_trace_metric.reset_states()
        test_field_metric.reset_states()
        test_intensity_metric.reset_states()


        # log metrics using wandb.log
        wandb.log({'Epochs': epoch,
                   'Train joint loss (trace MSE + intensity MSE)': np.mean(train_loss),
                   'Train trace MSE': float(train_trace_acc),
                   'Train field MSE': float(train_field_acc),
                   'Train intensity MSE': float(train_intensity_acc),
                   'Test joint loss (trace MSE + intensity MSE)': np.mean(val_loss),
                   'Test trace MSE': float(test_trace_acc),
                   'Test field MSE': float(test_field_acc),
                   'Test intensity MSE': float(test_intensity_acc)
                   })


def train_combined_loss_training(train_dataset, test_dataset, model, optimizer, trace_loss_fn, field_loss_fn, train_trace_metric, train_field_metric, test_trace_metric, test_field_metric, field_epochs, trace_epochs, start_with, reps, log_step, val_log_step, patience):
    """
    Training step that uses a combined training in which some steps are done with the trace loss function and
    others with the field loss function.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        model (tf.keras.Model): Model to train.
        optimizer (tf.keras.optimizers): Optimizer to use.
        trace_loss_fn (tf.keras.losses): Trace loss function to use.
        field_loss_fn (tf.keras.losses): Field loss function to use.
        train_trace_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the trace.
        train_field_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the field.
        test_trace_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the trace.
        test_field_metric (tf.keras.metrics): Accuracy metric that computes the MSe of the field.
        field_epochs (int): Number of epochs to train with the field loss function.
        trace_epochs (int): Number of epochs to train with the trace loss function.
        start_with (int): Start training with a certain loss function: 1 for trace, 0 for field.
        reps (int): Number of repetitions of the combined training.
        log_step (int): Number of steps to log training metrics.
        val_log_step (int): Number of steps to log validation metrics.
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
    """

    best_val_loss = float('inf')
    patience_counter = 0

    mode = start_with
    mode_epoch_counter = 0 # counter to change mode after a certain number of epochs

    for epoch in range((field_epochs + trace_epochs) * reps):

        if mode == 0:
            print("\nStart of total epoch %d with field loss epoch %d" % (epoch, mode_epoch_counter))

        if mode == 1:
            print("\nStart of total epoch %d with trace loss epoch %d" % (epoch, mode_epoch_counter))

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step_combined_loss_training(x_batch_train, y_batch_train,
                                    model, optimizer, mode, trace_loss_fn, field_loss_fn, train_trace_metric, train_field_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step_combined_loss_training(x_batch_val, y_batch_val,
                                        model, mode, trace_loss_fn, field_loss_fn, test_trace_metric, test_field_metric)
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
        train_trace_acc = train_trace_metric.result()
        train_field_acc = train_field_metric.result()
        print("Training trace acc over epoch: %.4e" % (float(train_trace_acc),))
        print("Training field acc over epoch: %.4e" % (float(train_field_acc),))

        test_trace_acc = test_trace_metric.result()
        test_field_acc = test_field_metric.result()
        print("Validation trace acc: %.4e" % (float(test_trace_acc),))
        print("Validation field acc: %.4e" % (float(test_field_acc),))

        # Reset metrics at the end of each epoch
        train_trace_metric.reset_states()
        train_field_metric.reset_states()
        test_trace_metric.reset_states()
        test_field_metric.reset_states()

        mode_epoch_counter += 1 # increase the counter of epochs with the same mode

        
        if mode == 1:
            if mode_epoch_counter >= trace_epochs:
                mode = 0
                mode_epoch_counter = 0
                print("Changing mode to field loss...")
        else:
            if mode_epoch_counter >= field_epochs:
                mode = 1
                mode_epoch_counter = 0
                print("Changing mode to trace loss...")

            

        # log metrics using wandb.log
        wandb.log({'Epochs': epoch,
                   'Train joint loss (trace MSE + field MSE)': np.mean(train_loss),
                   'Train trace MSE': float(train_trace_acc),
                   'Train field MSE': float(train_field_acc),
                   'Test joint loss (trace MSE + field MSE)': np.mean(val_loss),
                   'Test trace MSE': float(test_trace_acc),
                   'Test field MSE': float(test_field_acc)
                   })


def train_combined_loss_training_intensity(train_dataset, test_dataset, model, optimizer, trace_loss_fn, intensity_loss_fn, train_trace_metric, train_field_metric, train_intensity_metric, test_trace_metric, test_field_metric, test_intensity_metric, intensity_epochs, trace_epochs, start_with, reps, log_step, val_log_step, patience):
    """
    Training step that uses a combined training in which some steps are done with the trace loss function and
    others with the intensity loss function.

    Args:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
        model (tf.keras.Model): Model to train.
        optimizer (tf.keras.optimizers): Optimizer to use.
        trace_loss_fn (tf.keras.losses): Trace loss function to use.
        intensity_loss_fn (tf.keras.losses): Intensity loss function to use.
        train_trace_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the trace.
        train_field_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the field.
        train_intensity_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the intensity.
        test_trace_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the trace.
        test_field_metric (tf.keras.metrics): Accuracy metric that computes the MSe of the field.
        test_intensity_metric (tf.keras.metrics): Accuracy metric that computes the MSE of the intensity.
        intensity_epochs (int): Number of epochs to train with the field loss function.
        trace_epochs (int): Number of epochs to train with the trace loss function.
        start_with (int): Start training with a certain loss function: 1 for trace, 0 for field.
        reps (int): Number of repetitions of the combined training.
        log_step (int): Number of steps to log training metrics.
        val_log_step (int): Number of steps to log validation metrics.
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping.
    """

    best_val_loss = float('inf')
    patience_counter = 0

    mode = start_with
    mode_epoch_counter = 0 # counter to change mode after a certain number of epochs

    for epoch in range((intensity_epochs + trace_epochs) * reps):

        if mode == 0:
            print("\nStart of total epoch %d with intensity loss epoch %d" % (epoch, mode_epoch_counter))

        if mode == 1:
            print("\nStart of total epoch %d with trace loss epoch %d" % (epoch, mode_epoch_counter))

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step_combined_loss_training_intensity(x_batch_train, y_batch_train,
                                    model, optimizer, mode, trace_loss_fn, intensity_loss_fn, train_trace_metric, train_field_metric, train_intensity_metric)
            average_loss_value = tf.reduce_mean(loss_value)
            train_loss.append(float(average_loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(test_dataset):
            val_loss_value = test_step_combined_loss_training_intensity(x_batch_val, y_batch_val,
                                        model, mode, trace_loss_fn, intensity_loss_fn, test_trace_metric, test_field_metric, test_intensity_metric)
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
        train_trace_acc = train_trace_metric.result()
        train_field_acc = train_field_metric.result()
        train_intensity_acc = train_intensity_metric.result()
        print("Training trace acc over epoch: %.4e" % (float(train_trace_acc),))
        print("Training field acc over epoch: %.4e" % (float(train_field_acc),))
        print("Training intensity acc over epoch: %.4e" % (float(train_intensity_acc),))

        test_trace_acc = test_trace_metric.result()
        test_field_acc = test_field_metric.result()
        test_intensity_acc = test_intensity_metric.result()
        print("Validation trace acc: %.4e" % (float(test_trace_acc),))
        print("Validation field acc: %.4e" % (float(test_field_acc),))
        print("Validation intensity acc: %.4e" % (float(test_intensity_acc),))

        # Reset metrics at the end of each epoch
        train_trace_metric.reset_states()
        train_field_metric.reset_states()
        train_intensity_metric.reset_states()
        test_trace_metric.reset_states()
        test_field_metric.reset_states()
        test_intensity_metric.reset_states()

        mode_epoch_counter += 1 # increase the counter of epochs with the same mode
        
        if mode == 1:
            if mode_epoch_counter >= trace_epochs:
                mode = 0
                mode_epoch_counter = 0
                print("Changing mode to intensity loss...")
        else:
            if mode_epoch_counter >= intensity_epochs:
                mode = 1
                mode_epoch_counter = 0
                print("Changing mode to trace loss...")

            

        # log metrics using wandb.log
        wandb.log({'Epochs': epoch,
                   'Train joint loss (trace MSE + intensity MSE)': np.mean(train_loss),
                   'Train trace MSE': float(train_trace_acc),
                   'Train field MSE': float(train_field_acc),
                   'Train intensity MSE': float(train_intensity_acc),
                   'Test joint loss (trace MSE + intensity MSE)': np.mean(val_loss),
                   'Test trace MSE': float(test_trace_acc),
                   'Test field MSE': float(test_field_acc),
                   'Test intensity MSE': float(test_intensity_acc)
                   })