import tensorflow as tf

def train_step_MLP(x, y, model, optimizer, learning_rate, loss_fn, train_acc_metric):
    """
    Example training step for a MLP model.

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to use
        learning_rate (float): Learning rate to use
        loss_fn (tf.keras.losses.Loss): Loss function to use
        train_acc_metric (tf.keras.metrics.Metric): Metric to use for training accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the training step
    """
    with tf.GradientTape() as tape:
        results = model(x, training=True)
        loss_value = loss_fn(y, results)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, results)


    return loss_value