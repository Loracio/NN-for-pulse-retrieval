import tensorflow as tf

def custom_loss(y_true, y_pred):
    """
    Example custom loss function template.

    Args:
        y_true (tf.Tensor): True values
        y_pred (tf.Tensor): Predicted values

    Returns:
        tf.Tensor: Loss value
    """
    return tf.reduce_mean(tf.square(y_true - y_pred)) # Example loss function, MSE