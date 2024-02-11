import numpy as np
import tensorflow as tf

class fourier_utils():
    """
    Functions that involve Fourier Transforms for treating the SHG-FROG traces.

    They all share the same time and frequency arrays, as well as the phase factors,
    so memory is saved by computing them only once and storing them as attributes of the class.

    Args:
        N (int): Number of points in the trace
        Δt (float): Time step of the trace
    """
    def __init__(self, N, Δt):
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

    @tf.function
    def compute_trace(self, field_dataset):
        """
        Compute the trace of the signal operator given by the input electric field and the delayed electric field.

        Args:
            fields (tf.Tensor): Values of the electric field (real and imaginary parts concatenated in the last dimension of the tensor)

        Returns:
            tf.Tensor: Trace of the passed electric fields
        """

        # Take the real and imaginary parts of the predicted values
        fields_complex = tf.complex(field_dataset[:, :self.N], field_dataset[:, self.N:])

        # Compute the fourier transform of each predicted field
        # DTF(fields) = r_n · ifft(fields · s_j)
        # Where r_n is the phase factor of the fourier transform, fields is the
        # predicted electric field and s_j is the phase factor for delaying the
        # electric field
        predicted_spectrums = self.apply_DFT(fields_complex)

        # Delay each of the predicted spectrums by multiplying them by the delay factor
        delayed_predicted_spectrums = predicted_spectrums[:, None] * self.delay_factor

        # Bring back the delayed spectrums to the time domain
        delayed_predicted_pulses = self.apply_IDFT(delayed_predicted_spectrums)

        # Signal operator given by the predicted electric field and the delayed electric field
        signal_operator = fields_complex[:, None, :] * delayed_predicted_pulses

        # To obtain the trace of the signal operator, we need to compute the fourier transform
        fields_trace = tf.square(tf.abs(self.apply_DFT(signal_operator)))

        return fields_trace

    @tf.function
    def apply_DFT(self, x):
        """
        Apply the Discrete Fourier Transform to the input signal x.

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

        Args:
            x (tf.Tensor): Input signal to apply the IDFT to
        
        Returns:
            tf.Tensor: IDFT of the input signal
        """
        return self.s_j_conj[None, :] * tf.signal.fft(x * self.r_n_conj[None, :])

def meanVal(x, y):
    """
    Compute the mean value of x with respect to the probability distribution y.

    Args:
        x (np.array): values
        y (np.array): probability distribution

    Returns:
        mean (float): mean value of x
    """
    return np.sum(x * y) / np.sum(y)