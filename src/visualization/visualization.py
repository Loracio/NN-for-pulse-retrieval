import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backend_bases import Event

import tensorflow as tf

from ..io import load_and_norm_data, process_data
from ..utils import meanVal


class resultsGUI():
    def __init__(self, model, dataset, NUMBER_OF_PULSES, N, Δt, norm_predictions=True, phase_blanking=False, phase_blanking_threshold=1e-10):
        """
        This class implements a GUI for visualizing the results of the training process.

        Displays 5 different plots. At the top, the amplitude and phase of the input pulse
        in the time domain and in the frequency domain, superposed by the amplitude and phase
        of the output pulse in the time domain and in the frequency domain. At the bottom,
        the SHG FROG trace of the input pulse followed by the SHG FROG trace of the output pulse,
        then a plot of the difference between the trace of the output pulse and the input pulse.

        Args:
            model (tf.keras.Model): Trained model
            dataset (tf.data.Dataset): Dataset to visualize. It is prefetched and batched
            NUMBER_OF_PULSES (int): Number of pulses in the dataset
            N (int): Number of points in the trace
            Δt (float): Time step of the trace
            norm_predictions (bool): If True, the predicted fields are normalized
            phase_blanking (bool): If True, the phase of the predicted fields is blanked where the intensity of the true field is below a certain threshold
            phase_blanking_threshold (float): Threshold for blanking the phase of the predicted fields
        """

        self.model = model
        self.N = N
        self.Δt = Δt
        self.NUMBER_OF_PULSES = NUMBER_OF_PULSES

        self.norm_predictions = norm_predictions

        self.phase_blanking = phase_blanking
        self.phase_blanking_threshold = phase_blanking_threshold

        # The dataset is batched and prefetched. Now we don't need it to be batched
        # So we save the traces in a tensor and the true electric fields in another tensor
        self.true_traces = tf.TensorArray(tf.float32, size=NUMBER_OF_PULSES)
        self.true_electric_fields = tf.TensorArray(
            tf.complex64, size=NUMBER_OF_PULSES)

        index_to_write = 0
        for batch in dataset:
            for (trace, electric_field) in zip(batch[0], batch[1]):
                self.true_traces = self.true_traces.write(
                    index_to_write, trace)
                complex_electric_field = tf.complex(
                    electric_field[:N], electric_field[N:])
                self.true_electric_fields = self.true_electric_fields.write(
                    index_to_write, complex_electric_field)

                index_to_write += 1

        # Convert the TensorArray into a tensor
        self.true_traces = self.true_traces.stack()
        self.true_electric_fields = self.true_electric_fields.stack()

        # Evaluate the model with the true traces to obtain the predicted electric fields
        self.predicted_electric_fields = self.model(self.true_traces)
        # Convert into complex numbers
        self.predicted_electric_fields = tf.complex(
            self.predicted_electric_fields[:, :N], self.predicted_electric_fields[:, N:])

        # Initialize the fourier transform factors
        self.__init__fourier()

        # Get spectrums of the true and predicted electric fields
        self.true_spectrums = self.apply_DFT(self.true_electric_fields)
        self.predicted_spectrums = self.apply_DFT(
            self.predicted_electric_fields)

        # Compute the traces of the predicted electric fields
        self.predicted_traces = self.compute_trace(
            self.predicted_electric_fields)

        # Convert true electric fields and trace into numpy
        self.true_electric_fields = self.true_electric_fields.numpy()
        self.true_traces = self.true_traces.numpy()

        # Convert predicted electric fields and trace into numpy
        self.predicted_electric_fields = self.predicted_electric_fields.numpy()
        self.predicted_traces = self.predicted_traces.numpy()

        # Convert true and predicted spectrums into numpy
        self.true_spectrums = self.true_spectrums.numpy()
        self.predicted_spectrums = self.predicted_spectrums.numpy()

        # Compute trace errors
        self.compute_trace_errors()
        self.avg_trace_error = np.mean(self.trace_errors)

        # Compute field MSE errors
        self.compute_field_errors()
        self.avg_field_errors = np.mean(self.field_errors)

    def __init__fourier(self,):
        """
        Initialize the fourier transform factors
        """
        self.N_f = tf.cast(
            self.N, dtype=tf.float32)  # Number of points in the trace (float)
        self.Δt_f = tf.cast(self.Δt, dtype=tf.float32)  # Time step (float)
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

    def compute_field_errors(self,):
        """
        Compute a measure of the error of the predicted electric fields with respect to the true electric fields.
        A possible measure is the mean squared error of the absolute value of the difference between the true and predicted electric fields.
        """
        self.field_errors = []

        for i in range(self.NUMBER_OF_PULSES):
            self.field_errors.append(np.mean(
                np.abs(self.true_electric_fields[i] - self.predicted_electric_fields[i])**2))

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
        predicted_spectrums = self.r_n[None, :] * \
            tf.signal.ifft(y_pred_complex * self.s_j[None, :])

        # Delay each of the predicted spectrums by multiplying them by the delay factor
        delayed_predicted_spectrums = predicted_spectrums[:,
                                                          None] * self.delay_factor

        # Bring back the delayed spectrums to the time domain
        delayed_predicted_pulses = tf.map_fn(
            self.apply_IDFT, delayed_predicted_spectrums, dtype=tf.complex64)

        # Signal operator given by the predicted electric field and the delayed electric field
        signal_operator = y_pred_complex[:, None, :] * delayed_predicted_pulses

        # To obtain the trace of the signal operator, we need to compute the fourier transform
        y_pred_trace = tf.square(
            tf.abs(tf.map_fn(self.apply_DFT, signal_operator, dtype=tf.complex64)))

        return y_pred_trace

    def compute_trace_errors(self,):
        """
        Compute the trace errors of the predicted traces with respect to the true traces.

            R = (∑ₘₙ [Tₘₙᵐᵉᵃˢ - μ·Tₘₙ]²)½ / [N² (maxₘₙ Tₘₙᵐᵉᵃˢ)²]½

        Where μ is a scale factor given by:
            μ = ∑ₘₙ (Tₘₙᵐᵉᵃˢ · Tₘₙ) / (∑ₘₙ Tₘₙ²)

        And Tₘₙᵐᵉᵃˢ is the true trace, Tₘₙ is the predicted trace and m and n are the indices of the trace
        """
        self.trace_errors = []

        for i in range(self.NUMBER_OF_PULSES):
            # Compute the scale factor
            μ = np.sum(self.true_traces[i] * self.predicted_traces[i]
                       ) / np.sum(self.predicted_traces[i]**2)
            # Append trace error
            self.trace_errors.append(np.sqrt(np.sum(
                (self.true_traces[i] - μ * self.predicted_traces[i])**2) / (self.N**2 * np.max(self.true_traces[i])**2)))

    def plot(self,):

        self.fig, self.ax = plt.subplots()
        self.ax.set_axis_off()

        self.ax0 = plt.axes([.25/4, 0.1, 0.25, 0.35])
        self.ax1 = plt.axes([(.5/4 + 0.25), 0.1, 0.25, 0.35])
        self.ax2 = plt.axes([(1 - .25/4 - 0.25), 0.1, 0.25, 0.35])
        self.ax3 = plt.axes([(.5/4 + 0.25 - 0.025), 0.55, 0.25, 0.35])
        self.ax4 = plt.axes([(1 - .25/4 - 0.25), 0.55, 0.25, 0.35])
        self.twin_ax3 = self.ax3.twinx()
        self.twin_ax4 = self.ax4.twinx()

        plt.figtext(0.1, 0.7, 'Visualization control', fontweight='bold')
        self.next_button = Button(plt.axes(
            [0.15, 0.6, 0.075, 0.04]), 'Next', hovercolor='aquamarine', color='lightcyan')
        self.next_button.on_clicked(self.next_result)
        self.prev_button = Button(plt.axes(
            [0.05, 0.6, 0.075, 0.04]), 'Previous', hovercolor='aquamarine', color='lightcyan')
        self.prev_button.on_clicked(self.prev_result)

        plt.figtext(0.1, 0.9, 'Visualization mode', fontweight='bold')
        self.random_button = Button(plt.axes(
            [0.025, 0.8, 0.075, 0.04]), 'Random', hovercolor='aquamarine', color='mediumseagreen')
        self.random_button.on_clicked(self.display_random)
        self.best_button = Button(plt.axes(
            [0.125, 0.8, 0.075, 0.04]), 'Best', hovercolor='aquamarine', color='lightgrey')
        self.best_button.on_clicked(self.display_best)
        self.worst_button = Button(plt.axes(
            [0.225, 0.8, 0.075, 0.04]), 'Worst', hovercolor='aquamarine', color='lightgrey')
        self.worst_button.on_clicked(self.display_worst)

        self.best_index_order = np.argsort(self.trace_errors)  # Best to worst
        self.worst_index_order = np.flip(
            self.best_index_order)  # Worst to best
        self.random_index_order = np.random.permutation(
            self.NUMBER_OF_PULSES)  # Random order of the results

        self.last_index = 0
        self.previously_clicked = None
        self.mode = 'random'

        self.im0 = self.ax0.pcolormesh(self.omega, self.t, self.true_traces[0][:].reshape(
            self.N, self.N) / np.max(self.true_traces[0][:]), cmap='nipy_spectral')
        self.fig.colorbar(self.im0, ax=self.ax0)
        self.ax0.set_xlabel("Frequency")
        self.ax0.set_ylabel("Delay")
        self.ax0.set_title("True trace")

        self.im1 = self.ax1.pcolormesh(self.omega, self.t, self.predicted_traces[0].reshape(
            self.N, self.N) / np.max(self.predicted_traces[0][:]), cmap='nipy_spectral')
        self.fig.colorbar(self.im1, ax=self.ax1)
        self.ax1.set_xlabel("Frequency")
        self.ax1.set_ylabel("Delay")
        self.ax1.set_title("Predicted trace")

        self.im2 = self.ax2.pcolormesh(self.omega, self.t, np.abs(
            self.true_traces[0] - self.predicted_traces[0]).reshape(self.N, self.N), cmap='RdBu')
        self.colorbar2 = self.fig.colorbar(self.im2, ax=self.ax2)
        self.ax2.set_xlabel("Frequency")
        self.ax2.set_ylabel("Delay")
        self.ax2.set_title(r"$\sum_{mn} |T_{mn} - \hat{T}_{mn}|$")

        # Complex numbers to polar representation
        I_predicted_field = np.abs(self.predicted_electric_fields[0])**2
        I_predicted_spectrum = np.abs(self.predicted_spectrums[0])**2

        I_true_field = np.abs(self.true_electric_fields[0])**2
        I_true_spectrum = np.abs(self.true_spectrums[0])**2

        if self.norm_predictions:
            I_predicted_field = I_predicted_field / np.max(I_predicted_field)
            I_predicted_spectrum = I_predicted_spectrum / \
                np.max(I_predicted_spectrum)
            I_true_field = I_true_field / np.max(I_true_field)
            I_true_spectrum = I_true_spectrum / np.max(I_true_spectrum)

        predicted_field_phase = np.unwrap(
            np.angle(self.predicted_electric_fields[0]))
        predicted_field_phase -= meanVal(predicted_field_phase,
                                         I_predicted_field)

        predicted_spectrum_phase = np.unwrap(
            np.angle(self.predicted_spectrums[0]))
        predicted_spectrum_phase -= meanVal(
            predicted_spectrum_phase, I_predicted_spectrum)

        true_field_phase = np.unwrap(np.angle(self.true_electric_fields[0]))
        true_field_phase -= meanVal(true_field_phase, I_true_field)

        true_spectrum_phase = np.unwrap(np.angle(self.true_spectrums[0]))
        true_spectrum_phase -= meanVal(true_spectrum_phase, I_true_spectrum)

        if self.phase_blanking:
            predicted_field_phase = np.where(
                I_true_field < self.phase_blanking_threshold, np.nan, predicted_field_phase)
            predicted_spectrum_phase = np.where(
                I_true_spectrum < self.phase_blanking_threshold, np.nan, predicted_spectrum_phase)
            true_field_phase = np.where(
                I_true_field < self.phase_blanking_threshold, np.nan, true_field_phase)
            true_spectrum_phase = np.where(
                I_true_spectrum < self.phase_blanking_threshold, np.nan, true_spectrum_phase)

        self.line_I_true_field, = self.ax3.plot(
            self.t, I_true_field, color='blue', linewidth=3, alpha=0.5, label='True field intensity')
        self.line_true_field_phase, = self.twin_ax3.plot(
            self.t, true_field_phase, '-.', color='red', alpha=0.5)
        self.ax3.plot(np.nan, '-.', label='True phase', color='red')
        self.line_I_predicted_field, = self.ax3.plot(
            self.t, I_predicted_field, color='orange', label='Predicted field intensity')
        self.line_predicted_field_phase, = self.twin_ax3.plot(
            self.t, predicted_field_phase, '-.', color='violet')
        self.ax3.plot(np.nan, '-.', label='Predicted phase', color='violet')
        self.ax3.set_xlabel("Time")
        self.ax3.set_ylabel("Intensity")
        self.twin_ax3.set_ylabel("Phase")
        self.ax3.set_title("Time domain")
        self.ax3.grid()
        self.twin_ax3.set_ylim(-1.25 * 2 * np.pi, 1.25 * 2 * np.pi)

        self.line_I_true_spectrum, = self.ax4.plot(
            self.omega, I_true_spectrum, color='blue', linewidth=3, alpha=0.5, label='True spectral intensity')
        self.line_true_spectrum_phase, = self.twin_ax4.plot(
            self.omega, true_spectrum_phase, '-.', color='red', alpha=0.5)
        self.ax4.plot(np.nan, '-.', label='True spectral phase', color='red')
        self.line_I_predicted_spectrum, = self.ax4.plot(
            self.omega, I_predicted_spectrum, color='orange', label='Predicted spectral intensity')
        self.line_predicted_spectrum_phase, = self.twin_ax4.plot(
            self.omega, predicted_spectrum_phase, '-.', color='violet')
        self.ax4.plot(
            np.nan, '-.', label='Predicted spectral phase', color='violet')
        self.ax4.set_xlabel("Frequency")
        self.ax4.set_ylabel("Intensity")
        self.twin_ax4.set_ylabel("Phase")
        self.ax4.set_title("Frequency domain")
        self.ax4.grid()
        self.twin_ax4.set_ylim(-2 * np.pi, 2 * np.pi)

        self.fig.legend(*self.ax3.get_legend_handles_labels(),
                        loc='upper right', ncols=4)

        self.pulse_info_text = self.fig.text(0.02, 0.95, fr" Trace error = {self.format_scientific_notation(self.trace_errors[0])}  Avg trace error = {self.format_scientific_notation(self.avg_trace_error)}" +
                                             "\n" + fr"Field MSE = {self.format_scientific_notation(self.field_errors[0])}  Avg field MSE = {self.format_scientific_notation(self.avg_field_errors)}", fontweight='bold')

        return self.fig, self.ax

    def display_random(self, event):
        """
        Action of activating the button to display the results in random order
        """
        self.random_button.color = "mediumseagreen"
        self.worst_button.color = "lightgrey"
        self.best_button.color = "lightgrey"

        self.previously_clicked = None
        self.last_index = 0
        self.mode = 'random'

        self.next_result(Event('button_press_event', self.fig))

    def display_best(self, event):
        """
        Action of activating the button to display the results from best to worst
        """
        self.random_button.color = "lightgrey"
        self.worst_button.color = "lightgrey"
        self.best_button.color = "mediumseagreen"

        self.previously_clicked = None
        self.last_index = 0
        self.mode = 'best'

        self.next_result(Event('button_press_event', self.fig))

    def display_worst(self, event):
        """
        Action of activating the button to display the results from worst to best
        """
        self.random_button.color = "lightgrey"
        self.worst_button.color = "mediumseagreen"
        self.best_button.color = "lightgrey"

        self.previously_clicked = None
        self.last_index = 0
        self.mode = 'worst'

        self.next_result(Event('button_press_event', self.fig))

    def format_scientific_notation(self, value, precision=2):
        """
        Format a number in scientific notation to be displayed in a plot

        Args:
            value (float): Value to be formatted
            precision (int): Number of decimal places to display

        Returns:
            str: Formatted string
        """
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / 10**exponent
        formatted_str = f"${coefficient:.{precision}f} \\times 10^{{{exponent}}}$"
        return formatted_str

    def next_result(self, event):
        """
        Action of showing the next result according to the current mode.

        There is a lot of repeated code and it could be written in a cleaner way,
        but it is functional.
        """

        if self.previously_clicked == 'prev':
            self.last_index += 2

        self.update_plot()

        self.previously_clicked = 'next'
        self.last_index += 1

        plt.draw()

    def prev_result(self, event):
        """
        Action of showing the previous result according to the current mode.

        There is a lot of repeated code and it could be written in a cleaner way,
        but it is functional.
        """

        if self.previously_clicked == 'next':
            self.last_index -= 2

        self.update_plot()

        self.previously_clicked = 'prev'
        self.last_index -= 1

        plt.draw()

    def update_plot(self):
        """
        Update the plot with the current index and mode
        """

        match self.mode:
            case 'random': i = self.random_index_order[self.last_index]

            case 'best': i = self.best_index_order[self.last_index]

            case 'worst': i = self.worst_index_order[self.last_index]

        self.pulse_info_text.set_text(fr" Trace error = {self.format_scientific_notation(self.trace_errors[i])}  Avg trace error = {self.format_scientific_notation(self.avg_trace_error)}" +
                                      "\n" + fr"Field MSE = {self.format_scientific_notation(self.field_errors[i])}  Avg field MSE = {self.format_scientific_notation(self.avg_field_errors)}")

        self.im0.set_array(self.true_traces[i].reshape(
            self.N, self.N) / np.max(self.true_traces[i]))
        self.im0.set_clim(0, 1)

        max_Tpred = np.max(self.predicted_traces[i])
        if max_Tpred != 0:
            self.im1.set_array(self.predicted_traces[i].reshape(
                self.N, self.N) / max_Tpred)
            diff = np.abs(self.true_traces[i] / np.max(self.true_traces[i]) -
                          self.predicted_traces[i] / max_Tpred).reshape(self.N, self.N)
        else:
            self.im1.set_array(np.zeros((self.N, self.N)))
            diff = np.abs(
                self.true_traces[i] / np.max(self.true_traces[i])).reshape(self.N, self.N)

        self.im1.set_clim(0, 1)

        self.im2.set_array(diff)
        self.im2.set_clim(np.min(diff), np.max(diff))

        # Complex numbers to polar representation
        I_predicted_field = np.abs(self.predicted_electric_fields[i])**2
        I_predicted_spectrum = np.abs(self.predicted_spectrums[i])**2

        I_true_field = np.abs(self.true_electric_fields[i])**2
        I_true_spectrum = np.abs(self.true_spectrums[i])**2

        if self.norm_predictions:
            I_predicted_field = I_predicted_field / np.max(I_predicted_field)
            I_predicted_spectrum = I_predicted_spectrum / \
                np.max(I_predicted_spectrum)
            I_true_field = I_true_field / np.max(I_true_field)
            I_true_spectrum = I_true_spectrum / np.max(I_true_spectrum)

        predicted_field_phase = np.unwrap(
            np.angle(self.predicted_electric_fields[i]))
        predicted_field_phase -= meanVal(predicted_field_phase,
                                         I_predicted_field)

        predicted_spectrum_phase = np.unwrap(
            np.angle(self.predicted_spectrums[i]))
        predicted_spectrum_phase -= meanVal(
            predicted_spectrum_phase, I_predicted_spectrum)

        true_field_phase = np.unwrap(np.angle(self.true_electric_fields[i]))
        true_field_phase -= meanVal(true_field_phase, I_true_field)

        true_spectrum_phase = np.unwrap(np.angle(self.true_spectrums[i]))
        true_spectrum_phase -= meanVal(true_spectrum_phase, I_true_spectrum)

        if self.phase_blanking:
            predicted_field_phase = np.where(
                I_true_field < self.phase_blanking_threshold, np.nan, predicted_field_phase)
            predicted_spectrum_phase = np.where(
                I_true_spectrum < self.phase_blanking_threshold, np.nan, predicted_spectrum_phase)
            true_field_phase = np.where(
                I_true_field < self.phase_blanking_threshold, np.nan, true_field_phase)
            true_spectrum_phase = np.where(
                I_true_spectrum < self.phase_blanking_threshold, np.nan, true_spectrum_phase)

        self.line_I_predicted_field.set_ydata(I_predicted_field)
        self.line_predicted_field_phase.set_ydata(predicted_field_phase)
        self.line_I_predicted_spectrum.set_ydata(I_predicted_spectrum)
        self.line_predicted_spectrum_phase.set_ydata(predicted_spectrum_phase)

        self.line_I_true_field.set_ydata(I_true_field)
        self.line_true_field_phase.set_ydata(true_field_phase)
        self.line_I_true_spectrum.set_ydata(I_true_spectrum)
        self.line_true_spectrum_phase.set_ydata(true_spectrum_phase)
