import tensorflow as tf

@tf.function
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

    test_acc_metric.update_state(y, val_results)

    return loss_value

@tf.function
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

@tf.function
def test_step_MLP_intensity_loss(x, y, model, intensity_loss_fn, test_acc_metric):
    """
    Example test step for a MLP model with a intensity loss function (trace mse).

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        intensity_loss_fn (tf.keras.losses.Loss): Loss function to use
        test_acc_metric (tf.keras.metrics.Metric): Metric to use for test accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the validation step
    """
    val_results = model(x, training=False)
    loss_value = intensity_loss_fn(y, val_results)

    test_acc_metric.update_state(y, val_results)

    return loss_value

@tf.function
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

@tf.function
def test_step_joint_loss(x, y, model, weight_trace_loss, trace_loss_fn, weight_field_loss, mse_loss_fn, trace_acc_metric, field_acc_metric):
    """
    Example test step for a model with a joint loss function.

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to use
        weight_trace_loss (tf.Tensor): Weight for the trace loss
        trace_loss_fn (tf.keras.losses.Loss): Trace loss function to use
        weight_field_loss (tf.Tensor): Weight for the field loss
        mse_loss_fn (tf.keras.losses.Loss): MSE loss function to use
        trace_acc_metric (tf.keras.metrics.Metric): Metric to use for trace accuracy
        field_acc_metric (tf.keras.metrics.Metric): Metric to use for field accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the validation step
    """
    val_results = model(x, training=False)

    # Complex tensor of predictions
    N = val_results.shape[1] // 2
    y_pred_complex = tf.complex(val_results[:, :N], val_results[:, N:])

    # Normalize the predicted fields by the max value of each of them
    max_values = tf.reduce_max(tf.abs(y_pred_complex), axis=1, keepdims=True)
    results_normalized = val_results / max_values

    loss_value = weight_trace_loss * trace_loss_fn(x, val_results) + weight_field_loss * mse_loss_fn(y, results_normalized)

    trace_acc_metric.update_state(x, val_results)
    field_acc_metric.update_state(y, results_normalized)

    return loss_value

@tf.function
def test_step_joint_loss_intensity(x, y, model, weight_trace_loss, trace_loss_fn, weight_intensity_loss, mse_loss_fn, trace_acc_metric, field_acc_metric, intensity_acc_metric):
    """
    Example test step for a model with a joint loss function.

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to use
        weight_trace_loss (tf.Tensor): Weight for the trace loss
        trace_loss_fn (tf.keras.losses.Loss): Trace loss function to use
        weight_intensity_loss (tf.Tensor): Weight for the intensity loss
        mse_loss_fn (tf.keras.losses.Loss): MSE loss function to use
        trace_acc_metric (tf.keras.metrics.Metric): Metric to use for trace accuracy
        field_acc_metric (tf.keras.metrics.Metric): Metric to use for field accuracy
        intensity_acc_metric (tf.keras.metrics.Metric): Metric to use for intensity accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the validation step
    """
    val_results = model(x, training=False)

    # Complex tensor of predictions
    N = val_results.shape[1] // 2
    y_pred_complex = tf.complex(val_results[:, :N], val_results[:, N:])

    # Normalize the predicted fields by the max value of each of them
    max_values = tf.reduce_max(tf.abs(y_pred_complex), axis=1, keepdims=True)
    results_normalized = val_results / max_values

    loss_value = weight_trace_loss * trace_loss_fn(x, val_results) + weight_intensity_loss * mse_loss_fn(y, val_results)

    trace_acc_metric.update_state(x, val_results)
    field_acc_metric.update_state(y, results_normalized)
    intensity_acc_metric.update_state(y, val_results)

    return loss_value

@tf.function
def test_step_combined_loss_training(x, y, model, mode, trace_loss_fn, field_loss_fn, trace_acc_metric, field_acc_metric):
    """
    Test step of a combined training in which some steps are done with the trace loss function and
    others with the field loss function.

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        mode (int): Mode to use (1 for trace, 2 for field)
        loss_fn (tf.keras.losses.Loss): Loss function to use
        trace_acc_metric (tf.keras.metrics.Metric): Metric to use for trace accuracy
        field_acc_metric (tf.keras.metrics.Metric): Metric to use for field accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the validation step
    """

    val_results = model(x, training=False)

    # Complex tensor of predictions
    N = val_results.shape[1] // 2
    y_pred_complex = tf.complex(val_results[:, :N], val_results[:, N:])

    # Normalize the predicted fields by the max value of each of them
    max_values = tf.reduce_max(tf.abs(y_pred_complex), axis=1, keepdims=True)
    results_normalized = val_results / max_values

    loss_value = None

    if mode == 1:
        # Trace loss
        loss_value = trace_loss_fn(x, val_results)

    else:
        # Field loss
        loss_value = field_loss_fn(y, results_normalized)

    trace_acc_metric.update_state(x, val_results)
    field_acc_metric.update_state(y, results_normalized)

    return loss_value

@tf.function
def test_step_combined_loss_training_intensity(x, y, model, mode, trace_loss_fn, intensity_loss_fn, trace_acc_metric, field_acc_metric, intensity_acc_metric):
    """
    Test step of a combined training in which some steps are done with the trace loss function and
    others with the field loss function.

    Args:
        x (tf.Tensor): Input data
        y (tf.Tensor): Target data
        model (tf.keras.Model): Model to train
        mode (int): Mode to use (1 for trace, 2 for field)
        loss_fn (tf.keras.losses.Loss): Loss function to use
        trace_acc_metric (tf.keras.metrics.Metric): Metric to use for trace accuracy
        field_acc_metric (tf.keras.metrics.Metric): Metric to use for field accuracy
        intensity_acc_metric (tf.keras.metrics.Metric): Metric to use for intensity accuracy

    Returns:
        loss_value (tf.Tensor): Loss value for the validation step
    """

    val_results = model(x, training=False)

    # Complex tensor of predictions
    N = val_results.shape[1] // 2
    y_pred_complex = tf.complex(val_results[:, :N], val_results[:, N:])

    # Normalize the predicted fields by the max value of each of them
    max_values = tf.reduce_max(tf.abs(y_pred_complex), axis=1, keepdims=True)
    results_normalized = val_results / max_values

    loss_value = None

    if mode == 1:
        # Trace loss
        loss_value = trace_loss_fn(x, val_results)

    else:
        # Field loss
        loss_value = intensity_loss_fn(y, val_results)

    trace_acc_metric.update_state(x, val_results)
    field_acc_metric.update_state(y, results_normalized)
    intensity_acc_metric.update_state(y, val_results)

    return loss_value