import tensorflow as tf
import numpy as np

class trace_loss(tf.keras.losses.Loss):
    """
    Custom loss function to compute the MSE loss between the true and predicted values of the SHG-FROG traces (both normalized).

    loss = mean((y_true_normalized - y_pred_trace_normalized)^2)

    Where y_true_normalized is the true SHG-FROG trace normalized, y_pred_trace_normalized is the predicted SHG-FROG trace normalized
    and mean is the mean value of the tensor.

    Args:
        N (int): Number of points in the trace
        Δt (float): Time step of the trace
        name (str): Name of the loss function
    """
    def __init__(self, N, Δt, name="trace_loss"):
        super().__init__(name=name)
        self.N = N
        self.Δt = Δt

        self.N_f = tf.cast(N, dtype=tf.float32) # Number of points in the trace (float)
        self.Δt_f = tf.cast(Δt, dtype=tf.float32) # Time step (float)
        self.Δω_f = 2 * np.pi / (self.N_f * self.Δt_f) # Frequency step (float)

        self.t = tf.cast(tf.range(self.N), dtype=tf.float32) * self.Δt_f - self.N_f / 2 * self.Δt_f # Time array
        self.omega = - np.pi / self.Δt_f + tf.cast(tf.range(self.N), dtype=tf.float32) * self.Δω_f # Angular frequency array

        # Compute the phase factor of the fourier transform
        # rₙ = Δt / 2π · e^{i n t₀ Δω} ; sⱼ = e^{i ω₀ tⱼ}
        # Where n is the index of the time array, t₀ is the first time value, Δω is the
        # angular frequency step, j is the index of the angular frequency array, ω₀ is the
        # first angular frequency value and tⱼ is the j-th time value
        self.r_n = tf.exp(tf.complex(tf.cast(0.0, dtype=tf.float32), self.t[0] * self.Δω_f * tf.cast(tf.range(self.N), dtype=tf.float32)))
        self.r_n_conj = tf.math.conj(self.r_n)
        self.r_n = self.r_n * tf.complex(self.Δt_f / (2 * np.pi), 0.0) # Including amplitude factor to the phase factor (r_n) to avoid multiplying by it later
        self.s_j = tf.exp(tf.complex(0.0, self.omega[0] * self.t))
        self.s_j_conj = tf.complex(self.Δω_f, 0.0) * tf.math.conj(self.s_j) # Including Δω_f to the phase factor (s_j) to avoid multiplying by it later

        # For delaying each of the predicted electric fields, we need to multiply
        # each of the spectrums by a phase factor: exp(complex(0, omega[j] * t[i]))
        # where omega is the angular frequency array and t is the time array
        # i and j are the indices of the arrays, so we need to create a matrix
        # of shape (N, N) where each element is the product of the corresponding
        # elements of the arrays
        self.delay_factor = tf.exp(tf.complex(0.0, self.omega[None, :] * self.t[:, None]))

    def call(self, y_true, y_pred):
        """
        Compute the MSE loss between the true and predicted values of the SHG-FROG traces (both normalized).

        Args:
            y_true (tf.Tensor): True SHG-FROG trace
            y_pred (tf.Tensor): Predicted values of the electric field (real and imaginary parts concatenated in the last dimension of the tensor)

        Returns:
            tf.Tensor: MSE loss between the true and predicted values of the SHG-FROG traces    
        """
        # Take the real and imaginary parts of the predicted values
        # and construct a complex tensor that has the predicted electric field
        y_pred_real = y_pred[:, :self.N]
        y_pred_imag = y_pred[:, self.N:]
        y_pred_complex = tf.complex(y_pred_real, y_pred_imag)

        # Compute predicted traces for the input electric fields
        y_pred_trace = self.compute_trace(y_pred_complex)

        # Normalize the predicted and true traces
        max_values = tf.reduce_max(y_pred_trace, axis=[1, 2], keepdims=True)
        y_pred_trace_normalized = y_pred_trace / max_values

        max_values_true = tf.reduce_max(y_true, axis=[1, 2], keepdims=True)
        y_true_normalized = y_true / max_values_true

        # Compute the MSE loss
        loss = tf.reduce_mean((y_true_normalized - y_pred_trace_normalized)**2)

        return loss

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
        predicted_spectrums = self.r_n[None, :] * tf.signal.ifft(y_pred_complex * self.s_j[None, :])

        # Delay each of the predicted spectrums by multiplying them by the delay factor
        delayed_predicted_spectrums = predicted_spectrums[:, None] * self.delay_factor

        # Bring back the delayed spectrums to the time domain
        delayed_predicted_pulses = tf.map_fn(self.apply_IDFT, delayed_predicted_spectrums, dtype=tf.complex64)

        # Signal operator given by the predicted electric field and the delayed electric field
        signal_operator = y_pred_complex[:, None, :] * delayed_predicted_pulses

        # To obtain the trace of the signal operator, we need to compute the fourier transform
        y_pred_trace = tf.square(tf.abs(tf.map_fn(self.apply_DFT, signal_operator, dtype=tf.complex64)))

        return y_pred_trace