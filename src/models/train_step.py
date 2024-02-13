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

@tf.function
def train_step_joint_loss(x, y, model, optimizer, weight_trace_loss, trace_loss_fn, weight_field_loss, mse_loss_fn, trace_acc_metric, field_acc_metric):
    """
    Training step of a joint loss function (trace mse + mse).

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to use
        weight_trace_loss (float): Weight for the trace loss
        trace_loss_fn (tf.keras.losses.Loss): Trace loss function to use
        weight_field_loss (float): Weight for the field loss
        mse_loss_fn (tf.keras.losses.Loss): Field loss function to use
        train_acc_metric (tf.keras.metrics.Metric): Metric to use for training accuracy
    """
    with tf.GradientTape() as tape:
        results = model(x, training=True)
        # Complex tensor of predictions
        N = results.shape[1] // 2
        y_pred_complex = tf.complex(results[:, :N], results[:, N:])

        # Normalize the predicted fields by the max value of each of them
        max_values = tf.reduce_max(tf.abs(y_pred_complex), axis=1, keepdims=True)
        results_normalized = results / max_values

        loss_value = weight_trace_loss * trace_loss_fn(x, results) + weight_field_loss * mse_loss_fn(y, results_normalized)
        
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Update trace MSE metric
    trace_acc_metric.update_state(x, results)
    # Update field MSE metric
    field_acc_metric.update_state(y, results_normalized)


    return loss_value

@tf.function
def train_step_combined_loss_training(x, y, model, optimizer, mode, trace_loss_fn, field_loss_fn, trace_acc_metric, field_acc_metric):
    """
    Training step of a combined training in which some steps are done with the trace loss function and
    others with the field loss function.

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to use
        mode (str): Mode of training (1 train with trace, 0 train with field)
        loss_fn (tf.keras.losses.Loss): Loss function to use
        trace_acc_metric (tf.keras.metrics.Metric): Metric to use for training accuracy
        field_acc_metric (tf.keras.metrics.Metric): Metric to use for training accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the training step
    """
    with tf.GradientTape() as tape:
        results = model(x, training=True)
        
        # Complex tensor of predictions
        N = results.shape[1] // 2
        y_pred_complex = tf.complex(results[:, :N], results[:, N:])

        # Normalize the predicted fields by the max value of each of them
        max_values = tf.reduce_max(tf.abs(y_pred_complex), axis=1, keepdims=True)
        results_normalized = results / max_values

        loss_value = None
        
        if mode == 1:
            # Train with trace
            loss_value = trace_loss_fn(x, results)

        else:
            # Train with field
            loss_value = field_loss_fn(y, results_normalized)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Update trace MSE metric
    trace_acc_metric.update_state(x, results)
    # Update field MSE metric
    field_acc_metric.update_state(y, results_normalized)


    return loss_value