import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def custom_loss(y_true, y_pred, N=64):
    # Take the real and imaginary parts of the predicted values
    # and construct a complex tensor that has the predicted electric field
    y_pred_real = y_pred[:, :N]
    y_pred_imag = y_pred[:, N:]
    y_pred_complex = tf.complex(y_pred_real, y_pred_imag)

    # Define the fourier transform parameters
    # All types need to be converted to be suitable with tensorflow
    N_f = tf.cast(N, dtype=tf.float32) # Number of points in the trace (float)
    DELTA_T_f = 1 / N_f # Time step (float)
    DELTA_OMEGA_f = 2 * np.pi / (N_f * DELTA_T_f) # Frequency step (float)

    # Time array, defined as i * DELTA_T, where i is the index of the time array
    # and DELTA_T is the time step
    # t = (i - N/2) * Δt
    t = tf.cast(tf.range(N), dtype=tf.float32) * DELTA_T_f - N_f / 2 * DELTA_T_f
    # Angular frequency array, defined as -PI / deltaT + (2 * PI * i / (N * deltaT)),
    # where i is the index of the angular frequency array, N is the number of points
    # in the trace and deltaT is the time step
    omega = - np.pi / DELTA_T_f + tf.cast(tf.range(N), dtype=tf.float32) * DELTA_OMEGA_f

    # Compute the phase factor of the fourier transform
    # rₙ = Δt / 2π · e^{i n t₀ Δω} ; sⱼ = e^{i ω₀ tⱼ}
    # Where n is the index of the time array, t₀ is the first time value, Δω is the
    # angular frequency step, j is the index of the angular frequency array, ω₀ is the
    # first angular frequency value and tⱼ is the j-th time value
    r_n = tf.exp(tf.complex(tf.cast(0.0, dtype=tf.float32), t[0] * DELTA_OMEGA_f * tf.cast(tf.range(N), dtype=tf.float32)))
    r_n_conj = tf.math.conj(r_n)
    r_n = r_n * tf.complex(DELTA_T_f / (2 * np.pi), 0.0)
    s_n = tf.exp(tf.complex(0.0, omega[0] * t))
    s_n_conj = tf.complex(DELTA_OMEGA_f, 0.0) * tf.math.conj(s_n)

    # For delaying each of the predicted electric fields, we need to multiply
    # each of the spectrums by a phase factor: exp(complex(0, omega[j] * t[i]))
    # where omega is the angular frequency array and t is the time array
    # i and j are the indices of the arrays, so we need to create a matrix
    # of shape (N, N) where each element is the product of the corresponding
    # elements of the arrays
    delay_factor = tf.exp(tf.complex(0.0, omega[None, :] * t[:, None]))

    # Compute the fourier transform of each predicted field
    # DTF(y_pred) = r_n · fft(y_pred · s_n)
    # Where r_n is the phase factor of the fourier transform, y_pred is the
    # predicted electric field and s_n is the phase factor for delaying the
    # electric field
    predicted_spectrums = r_n[None, :] * tf.signal.ifft(y_pred_complex * s_n[None, :])
    # Delay each of the predicted spectrums by multiplying them by the delay factor
    delayed_predicted_spectrums = predicted_spectrums[:, None] * delay_factor
    # Bring back the delayed spectrums to the time domain
    delayed_predicted_pulses = tf.map_fn(apply_IDFT, delayed_predicted_spectrums, dtype=tf.complex64)
    # Signal operator given by the predicted electric field and the delayed electric field
    signal_operator = y_pred_complex[:, None, :] * delayed_predicted_pulses
    # To obtain the trace of the signal operator, we need to compute the fourier transform
    y_pred_trace = tf.square(tf.abs(tf.map_fn(apply_DFT, signal_operator, dtype=tf.complex64)))

    # y_pred_trace = tf.square(tf.abs(signal_operator))
    max_values = tf.reduce_max(y_pred_trace, axis=[1, 2], keepdims=True)
    y_pred_trace_normalized = y_pred_trace / max_values

    max_values_true = tf.reduce_max(y_true, axis=[1, 2], keepdims=True)
    y_true_normalized = y_true / max_values_true

    # #! Plot each trace
    # for i in range(32):
    #     plt.figure()
    #     plt.imshow(y_pred_trace_normalized[i])
    #     plt.colorbar()

    #     # Plot the true value
    #     plt.figure()
    #     plt.imshow(y_true_normalized[i])
    #     plt.colorbar()

    #     plt.show()
    # #!

    # Compute the MSE loss
    loss = tf.reduce_mean((y_true_normalized - y_pred_trace_normalized)**2)

    return loss

def apply_IDFT(x, N=64):
    N_f = tf.cast(N, dtype=tf.float32) # Number of points in the trace (float)
    DELTA_T_f = 1 / N_f # Time step (float)
    DELTA_OMEGA_f = 2 * np.pi / (N_f * DELTA_T_f) # Frequency step (float)
    # Time array, defined as i * DELTA_T, where i is the index of the time array
    # and DELTA_T is the time step
    # t = (i - N/2) * Δt
    t = tf.cast(tf.range(N), dtype=tf.float32) * DELTA_T_f - N_f / 2 * DELTA_T_f
    # Angular frequency array, defined as -PI / deltaT + (2 * PI * i / (N * deltaT)),
    # where i is the index of the angular frequency array, N is the number of points
    # in the trace and deltaT is the time step
    omega = - np.pi / DELTA_T_f + tf.cast(tf.range(N), dtype=tf.float32) * DELTA_OMEGA_f

    # Compute the phase factor of the fourier transform
    # rₙ = Δt / 2π · e^{i n t₀ Δω} ; sⱼ = e^{i ω₀ tⱼ}
    # Where n is the index of the time array, t₀ is the first time value, Δω is the
    # angular frequency step, j is the index of the angular frequency array, ω₀ is the
    # first angular frequency value and tⱼ is the j-th time value
    r_n = tf.exp(tf.complex(tf.cast(0.0, dtype=tf.float32), t[0] * DELTA_OMEGA_f * tf.cast(tf.range(N), dtype=tf.float32)))
    r_n_conj = tf.math.conj(r_n)
    r_n = r_n * tf.complex(DELTA_T_f / (2 * np.pi), 0.0)
    s_n = tf.exp(tf.complex(0.0, omega[0] * t))
    s_n_conj = tf.complex(DELTA_OMEGA_f, 0.0) * tf.math.conj(s_n)
    return s_n_conj[None, :] * tf.signal.fft(x * r_n_conj[None, :])

def apply_DFT(x, N=64):
    N_f = tf.cast(N, dtype=tf.float32) # Number of points in the trace (float)
    DELTA_T_f = 1 / N_f # Time step (float)
    DELTA_OMEGA_f = 2 * np.pi / (N_f * DELTA_T_f) # Frequency step (float)
    # Time array, defined as i * DELTA_T, where i is the index of the time array
    # and DELTA_T is the time step
    # t = (i - N/2) * Δt
    t = tf.cast(tf.range(N), dtype=tf.float32) * DELTA_T_f - N_f / 2 * DELTA_T_f
    # Angular frequency array, defined as -PI / deltaT + (2 * PI * i / (N * deltaT)),
    # where i is the index of the angular frequency array, N is the number of points
    # in the trace and deltaT is the time step
    omega = - np.pi / DELTA_T_f + tf.cast(tf.range(N), dtype=tf.float32) * DELTA_OMEGA_f

    # Compute the phase factor of the fourier transform
    # rₙ = Δt / 2π · e^{i n t₀ Δω} ; sⱼ = e^{i ω₀ tⱼ}
    # Where n is the index of the time array, t₀ is the first time value, Δω is the
    # angular frequency step, j is the index of the angular frequency array, ω₀ is the
    # first angular frequency value and tⱼ is the j-th time value
    r_n = tf.exp(tf.complex(tf.cast(0.0, dtype=tf.float32), t[0] * DELTA_OMEGA_f * tf.cast(tf.range(N), dtype=tf.float32)))
    r_n_conj = tf.math.conj(r_n)
    r_n = r_n * tf.complex(DELTA_T_f / (2 * np.pi), 0.0)
    s_n = tf.exp(tf.complex(0.0, omega[0] * t))
    s_n_conj = tf.complex(DELTA_OMEGA_f, 0.0) * tf.math.conj(s_n)
    return r_n[None, :] * tf.signal.ifft(x * s_n[None, :])