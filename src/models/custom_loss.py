import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# TODO: Fix this function. The phase factors in the fft are lacking
def custom_loss(y_true, y_pred, N=64):
    """
    Custom loss function for the MLP and CNN models.
    The idea is to take the output of the NN, which is a 2N-dimensional vector containing
    the real and imaginary parts of the pulse in the time domain, and transform it into
    the SHG-FROG trace of the pulse, which is an NxN matrix. Then, we can compare this
    matrix with the SHG-FROG trace of the input pulse and calculate the error.

    Args:
        y_true (tf.Tensor): True values
        y_pred (tf.Tensor): Predicted values

    Returns:
        tf.Tensor: Loss value
    """

    y_pred_real = y_pred[:, :N]
    y_pred_imag = y_pred[:, N:]
    y_pred_complex = tf.complex(y_pred_real, y_pred_imag)

    # Compute the Fourier transform
    y_pred_complex_fft = tf.signal.fft(y_pred_complex)

    # Compute the phase factor for each tau
    N_f = tf.cast(N, dtype=tf.float32)
    τ = -tf.floor(0.5 * N_f) / N_f + tf.range(N, dtype=tf.float32) / N_f
    ω = - N_f / 2 + tf.range(N, dtype=tf.float32)
    ω *= 2 * np.pi
    Δω = 2 * np.pi * N_f
    phase_factor = tf.exp(tf.complex(0.0, ω[None, :] * τ[:, None]))

    # Compute the product of y_pred_complex_fft and phase_factor
    product = y_pred_complex_fft[:, None, :] * phase_factor

    # Compute the inverse Fourier transform
    inverse_fft = tf.signal.ifft(product)

    # Compute the product
    product = tf.TensorArray(dtype=tf.complex64, size=N)

    for i in range(N):
        product = product.write(i, y_pred_complex * inverse_fft[:, :, i])

    product = tf.transpose(product.stack(), perm=[1, 2, 0])

    # Compute the predicted SHG-FROG trace, shifting the fft
    y_pred_trace = tf.abs(tf.signal.fftshift(tf.signal.fft(product), axes=2))**2

    # #! Plot one of this traces
    # plt.figure()
    # plt.imshow(y_pred_trace[0])
    # plt.colorbar()
    # # plt.show()

    # # Plot the true value
    # plt.figure()
    # plt.imshow(y_true[0])
    # plt.colorbar()
    # plt.show()
    # #!

    # Compute the MSE loss
    loss = tf.reduce_mean((y_true - y_pred_trace)**2)

    return loss