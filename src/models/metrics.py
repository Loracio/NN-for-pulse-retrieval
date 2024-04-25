import tensorflow as tf
import numpy as np


class trace_MSE(tf.keras.metrics.Metric):
    """
    Custom metric to compute the trace MSE error between the true and predicted SHG-FROG traces (both normalized).

        Trace_MSE = mean((y_true_normalized - y_pred_trace_normalized)^2)

    Args:
        N (int): Number of points in the trace
        Δt (float): Time step of the trace
        name (str, optional): Name of the metric. Defaults to "trace_MSE".

    Returns:
        tf.Tensor: Trace MSE error between the true and predicted SHG-FROG traces
    """

    def __init__(self, N, Δt, name="trace_MSE", **kwargs):
        super(trace_MSE, self).__init__(name=name, **kwargs)

        self.N = N
        self.Δt = Δt

        # Number of points in the trace (float)
        self.N_f = tf.cast(N, dtype=tf.float32)
        self.Δt_f = tf.cast(Δt, dtype=tf.float32)  # Time step (float)
        # Frequency step (float)
        self.Δω_f = 2 * np.pi / (self.N_f * self.Δt_f)

        self.t = tf.cast(tf.range(self.N), dtype=tf.float32) * \
            self.Δt_f - self.N_f / 2 * self.Δt_f  # Time array
        self.omega = - np.pi / self.Δt_f + \
            tf.cast(tf.range(self.N), dtype=tf.float32) * \
            self.Δω_f  # Angular frequency array

        # Compute the phase factor of the fourier transform
        # rₙ = Δt / 2π · e^{i n t₀ Δω} ; sⱼ = e^{i ω₀ tⱼ}
        # Where n is the index of the time array, t₀ is the first time value, Δω is the
        # angular frequency step, j is the index of the angular frequency array, ω₀ is the
        # first angular frequency value and tⱼ is the j-th time value
        self.r_n = tf.exp(tf.complex(tf.cast(0.0, dtype=tf.float32),
                          self.t[0] * self.Δω_f * tf.cast(tf.range(self.N), dtype=tf.float32)))
        self.r_n_conj = tf.math.conj(self.r_n)
        # Including amplitude factor to the phase factor (r_n) to avoid multiplying by it later
        self.r_n = self.r_n * tf.complex(self.Δt_f / (2 * np.pi), 0.0)
        self.s_j = tf.exp(tf.complex(0.0, self.omega[0] * self.t))
        # Including Δω_f to the phase factor (s_j) to avoid multiplying by it later
        self.s_j_conj = tf.complex(self.Δω_f, 0.0) * tf.math.conj(self.s_j)

        # For delaying each of the predicted electric fields, we need to multiply
        # each of the spectrums by a phase factor: exp(complex(0, omega[j] * t[i]))
        # where omega is the angular frequency array and t is the time array
        # i and j are the indices of the arrays, so we need to create a matrix
        # of shape (N, N) where each element is the product of the corresponding
        # elements of the arrays
        self.delay_factor = tf.exp(tf.complex(
            0.0, self.omega[None, :] * self.t[:, None]))

        # Initialize the state of the metric
        self.trace_MSE = self.add_weight(
            name="trace_MSE", initializer="zeros")  # Trace error
        self.total = self.add_weight(
            name="total", initializer="zeros")  # Total number of samples

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the metric with the new data.

        Args:
            y_true (tf.Tensor): True SHG-FROG traces
            y_pred (tf.Tensor): Predicted values of the electric field (real and imaginary parts concatenated in the last dimension of the tensor)
            sample_weight (tf.Tensor, optional): Sample weights. Defaults to None.
        """
        # Take the real and imaginary parts of the predicted values
        # and construct a complex tensor that has the predicted electric field
        y_pred_complex = tf.complex(y_pred[:, :self.N], y_pred[:, self.N:])

        # Compute predicted traces for the input electric fields
        y_pred_trace = self.compute_trace(y_pred_complex)

        # Normalize the predicted and true traces
        max_values = tf.reduce_max(y_pred_trace, axis=[1, 2], keepdims=True)
        y_pred_trace_normalized = y_pred_trace / max_values

        self.trace_MSE.assign_add(tf.reduce_sum(tf.reduce_mean((y_true - y_pred_trace_normalized)**2)))  # Increase the trace error
        self.total.assign_add(1)  # Increase the total number of samples

    @tf.function
    def result(self):
        return self.trace_MSE / self.total

    @tf.function
    def reset_states(self):
        self.trace_MSE.assign(0.0)
        self.total.assign(0.0)

    @tf.function
    def apply_DFT(self, x):
        """
        Apply the Discrete Fourier Transform to the input signal x.
        This function is thought to be used with tf.map_fn to apply the DFT to a batch of signals.

        Args:
            x (tf.Tensor): Input signal to apply the DFT to

        Returns:
            tf.Tensor: DFT of the input signal
        """
        return self.r_n[None, :] * tf.signal.ifft(x * self.s_j[None, :])

    @tf.function
    def apply_IDFT(self, x):
        """
        Apply the Inverse Discrete Fourier Transform to the input signal x.
        This function is thought to be used with tf.map_fn to apply the IDFT to a batch of signals.

        Args:
            x (tf.Tensor): Input signal to apply the IDFT to

        Returns:
            tf.Tensor: IDFT of the input signal
        """
        return self.s_j_conj[None, :] * tf.signal.fft(x * self.r_n_conj[None, :])

    @tf.function
    def compute_trace(self, y_pred_complex):
        """
        Compute the trace of the signal operator given by the input electric field and the delayed electric field.

        Args:
            y_pred (tf.Tensor): Predicted values of the electric field (real and imaginary parts concatenated in the last dimension of the tensor)

        Returns:
            tf.Tensor: Trace of the passed electric fields
        """
        # Compute the fourier transform of each predicted field
        # DTF(y_pred) = r_n · ifft(y_pred · s_j)
        # Where r_n is the phase factor of the fourier transform, y_pred is the
        # predicted electric field and s_j is the phase factor for delaying the
        # electric field
        predicted_spectrums = self.apply_DFT(y_pred_complex)

        # Delay each of the predicted spectrums by multiplying them by the delay factor
        delayed_predicted_spectrums = predicted_spectrums[:, None] * self.delay_factor

        # Bring back the delayed spectrums to the time domain
        delayed_predicted_pulses = self.apply_IDFT(delayed_predicted_spectrums)

        # Signal operator given by the predicted electric field and the delayed electric field
        signal_operator = y_pred_complex[:, None, :] * delayed_predicted_pulses

        # To obtain the trace of the signal operator, we need to compute the fourier transform
        y_pred_trace = tf.square(tf.abs(self.apply_DFT(signal_operator)))

        return y_pred_trace


class intensity_MSE(tf.keras.metrics.Metric):
    """
    Custom metric to compute the MSE loss between the true and predicted values of the intensity in the time domain and frequency domain.

        Intensity_MSE = mean((I_true_time_normalized - I_pred_time_normalized)^2) + mean((I_true_freq_normalized - I_pred_freq_normalized)^2)

    Args:
        N (int): Number of points in the field
        Δt (float): Time step of the field
        name (str, optional): Name of the metric. Defaults to "intensity_MSE".

    Returns:
        tf.Tensor: intensity MSE error between the true and predicted fields
    """

    def __init__(self, N, Δt, name="intensity_MSE", **kwargs):
        super(intensity_MSE, self).__init__(name=name, **kwargs)

        self.N = N
        self.Δt = Δt

        # Number of points in the trace (float)
        self.N_f = tf.cast(N, dtype=tf.float32)
        self.Δt_f = tf.cast(Δt, dtype=tf.float32)  # Time step (float)
        # Frequency step (float)
        self.Δω_f = 2 * np.pi / (self.N_f * self.Δt_f)

        self.t = tf.cast(tf.range(self.N), dtype=tf.float32) * \
            self.Δt_f - self.N_f / 2 * self.Δt_f  # Time array
        self.omega = - np.pi / self.Δt_f + \
            tf.cast(tf.range(self.N), dtype=tf.float32) * \
            self.Δω_f  # Angular frequency array

        # Compute the phase factor of the fourier transform
        # rₙ = Δt / 2π · e^{i n t₀ Δω} ; sⱼ = e^{i ω₀ tⱼ}
        # Where n is the index of the time array, t₀ is the first time value, Δω is the
        # angular frequency step, j is the index of the angular frequency array, ω₀ is the
        # first angular frequency value and tⱼ is the j-th time value
        self.r_n = tf.exp(tf.complex(tf.cast(0.0, dtype=tf.float32),
                          self.t[0] * self.Δω_f * tf.cast(tf.range(self.N), dtype=tf.float32)))
        self.r_n_conj = tf.math.conj(self.r_n)
        # Including amplitude factor to the phase factor (r_n) to avoid multiplying by it later
        self.r_n = self.r_n * tf.complex(self.Δt_f / (2 * np.pi), 0.0)
        self.s_j = tf.exp(tf.complex(0.0, self.omega[0] * self.t))
        # Including Δω_f to the phase factor (s_j) to avoid multiplying by it later
        self.s_j_conj = tf.complex(self.Δω_f, 0.0) * tf.math.conj(self.s_j)

        # For delaying each of the predicted electric fields, we need to multiply
        # each of the spectrums by a phase factor: exp(complex(0, omega[j] * t[i]))
        # where omega is the angular frequency array and t is the time array
        # i and j are the indices of the arrays, so we need to create a matrix
        # of shape (N, N) where each element is the product of the corresponding
        # elements of the arrays
        self.delay_factor = tf.exp(tf.complex(
            0.0, self.omega[None, :] * self.t[:, None]))

        # Initialize the state of the metric
        self.intensity_MSE = self.add_weight(
            name="intensity_MSE", initializer="zeros")  # Intensity error
        self.total = self.add_weight(
            name="total", initializer="zeros")  # Total number of samples

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the metric with the new data.

        Args:
            y_true (tf.Tensor): True field (real and imaginary parts concatenated in the last dimension of the tensor)
            y_pred (tf.Tensor): Predicted values of the electric field (real and imaginary parts concatenated in the last dimension of the tensor)
            sample_weight (tf.Tensor, optional): Sample weights. Defaults to None.
        """
        # Take the real and imaginary parts of the predicted values and true values
        # and construct a complex tensor that has the predicted electric field
        y_pred_complex = tf.complex(y_pred[:, :self.N], y_pred[:, self.N:])
        max_values = tf.reduce_max(tf.abs(y_pred_complex), axis=1, keepdims=True)
        max_values = tf.cast(max_values, tf.complex64)
        y_pred_complex = y_pred_complex / max_values

        y_true_complex = tf.complex(y_true[:, :self.N], y_true[:, self.N:])

        # Compute intensities
        I_true_time = tf.abs(y_true_complex)
        I_pred_time = tf.abs(y_pred_complex)

        # Compute the spectrums of true and predicted values
        I_true_freq = self.apply_DFT(y_true_complex)
        I_pred_freq = self.apply_DFT(y_pred_complex)

        # Normalize the spectrums
        max_values = tf.reduce_max(tf.abs(I_true_freq), axis=1, keepdims=True)
        max_values = tf.cast(max_values, tf.complex64)
        I_true_freq = I_true_freq / max_values
        max_values = tf.reduce_max(tf.abs(I_pred_freq), axis=1, keepdims=True)
        max_values = tf.cast(max_values, tf.complex64)
        I_pred_freq = I_pred_freq / max_values

        # Compute intensities and convert to float32
        I_true_freq = tf.abs(I_true_freq)
        I_pred_freq = tf.abs(I_pred_freq)

        self.intensity_MSE.assign_add(tf.reduce_sum((I_true_time - I_pred_time)**2) + tf.reduce_sum((I_true_freq - I_pred_freq)**2))  # Increase the intensity error
        self.total.assign_add(1)  # Increase the total number of samples

    @tf.function
    def result(self):
        return self.intensity_MSE / self.total

    @tf.function
    def reset_states(self):
        self.intensity_MSE.assign(0.0)
        self.total.assign(0.0)

    @tf.function
    def apply_DFT(self, x):
        """
        Apply the Discrete Fourier Transform to the input signal x.
        This function is thought to be used with tf.map_fn to apply the DFT to a batch of signals.

        Args:
            x (tf.Tensor): Input signal to apply the DFT to

        Returns:
            tf.Tensor: DFT of the input signal
        """
        return self.r_n[None, :] * tf.signal.ifft(x * self.s_j[None, :])