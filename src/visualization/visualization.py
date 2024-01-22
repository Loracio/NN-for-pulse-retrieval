import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backend_bases import Event

import tensorflow as tf

from ..io import load_and_norm_data, process_data

# TODO: Implement this class
class resultsGUI():
    def __init__(self, t, Δt, N, NUMBER_OF_PULSES, FILE_PATH, MODEL_PATH, conv=False):
        """
        This class implements a GUI for visualizing the results of the training process.
        It is intended to be used with the MLP and CNN models.

        Displays 5 different plots. At the top, the amplitude and phase of the input pulse
        in the time domain and in the frequency domain, superposed by the amplitude and phase
        of the output pulse in the time domain and in the frequency domain. At the bottom,
        the SHG FROG trace of the input pulse followed by the SHG FROG trace of the output pulse,
        then a plot of the difference between the trace of the output pulse and the input pulse.
        Args:
            t (np.array): Time vector.
            Δt (float): Time step.
            N (int): Number of points in the input pulse.
            NUMBER_OF_PULSES (int): Number of pulses in the pulse database.
            FILE_PATH (str): Path to the pulse database.
            MODEL_PATH (str): Path to the model.
            conv (bool, optional): If True, the model is a CNN. Defaults to False.
        """

        self.t = t
        self.Δt = Δt
        self.N = N
        self.NUMBER_OF_PULSES = NUMBER_OF_PULSES

        self.frequencies = -1 / (2 * self.Δt) + np.arange(self.N) /(self.N * self.Δt)
        self.ω = self.frequencies *  2 * np.pi
        self.Δω = 2 * np.pi / self.Δt

        self.pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)

        self.model = tf.keras.models.load_model(MODEL_PATH)