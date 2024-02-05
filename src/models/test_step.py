import tensorflow as tf

def test_step_MLP(x, y, model, loss_fn, test_acc_metric):
    """
    Example test step for a MLP model.

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        loss_fn (tf.keras.losses.Loss): Loss function to use
        test_acc_metric (tf.keras.metrics.Metric): Metric to use for test accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the validation step
    """
    val_results = model(x, training=False)
    loss_value = loss_fn(y, val_results)

    test_acc_metric.update_state(x, val_results) # Trace MSE metric is called with the input data

    return loss_value

def test_step_MLP_custom_loss(x, y, model, custom_loss_fn, test_acc_metric):
    """
    Example test step for a MLP model with a custom loss function (trace mse).

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        custom_loss_fn (tf.keras.losses.Loss): Loss function to use
        test_acc_metric (tf.keras.metrics.Metric): Metric to use for test accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the validation step
    """
    val_results = model(x, training=False)
    loss_value = custom_loss_fn(x, val_results) # Note that the loss function is called with the input data

    test_acc_metric.update_state(y, val_results)

    return loss_value

def test_step_CNN_custom_loss(x, y, model, custom_loss_fn, test_acc_metric):
    """
    Example test step for a CNN model with a custom loss function (trace mse).

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        custom_loss_fn (tf.keras.losses.Loss): Loss function to use
        test_acc_metric (tf.keras.metrics.Metric): Metric to use for test accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the validation step
    """
    val_results = model(x, training=False)
    loss_value = custom_loss_fn(x, val_results) # Note that the loss function is called with the input data

    test_acc_metric.update_state(y, val_results)

    return loss_value