import tensorflow as tf

def train_step_MLP(x, y, model, optimizer, loss_fn, train_acc_metric):
    """
    Example training step for a MLP model.

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to use
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

def train_step_MLP_custom_loss(x, y, model, optimizer, custom_loss_fn, train_acc_metric):
    """
    Example training step for a MLP model with a custom loss function (trace mse).

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to use
        custom_loss_fn (tf.keras.losses.Loss): Loss function to use
        train_acc_metric (tf.keras.metrics.Metric): Metric to use for training accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the training step
    """
    with tf.GradientTape() as tape:
        results = model(x, training=True)
        loss_value = custom_loss_fn(x, results) # Note that the loss function is called with the input data
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, results)


    return loss_value

@tf.function
def train_step_CNN_custom_loss(x, y, model, optimizer, custom_loss_fn, train_acc_metric):
    """
    Example training step for a CNN model with a custom loss function (trace mse).

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to use
        custom_loss_fn (tf.keras.losses.Loss): Loss function to use
        train_acc_metric (tf.keras.metrics.Metric): Metric to use for training accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the training step
    """
    with tf.GradientTape() as tape:
        results = model(x, training=True)
        loss_value = custom_loss_fn(x, results) # Note that the loss function is called with the input data
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, results)


    return loss_value