import tensorflow as tf
from .train_step import train_step
from .test_step import test_step

def train_MLP(train_dataset, test_dataset, model, optimizer, learning_rate, loss_fn, train_acc_metric, test_acc_metric, epochs=EPOCHS, log_step=LOG_STEP, val_log_step=VAL_LOG_STEP, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step_MLP(x_batch_train, y_batch_train,
                                    model, optimizer, learning_rate,
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
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc),
                   'test_loss': np.mean(val_loss),
                   'test_acc': float(test_acc)
                   })