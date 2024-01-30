import unittest
import tensorflow as tf

import path_helper

from src.models.custom_loss import custom_loss
from src.io import load_and_norm_data, process_data

class TestCustomLoss(unittest.TestCase):
    def test_custom_loss(self):
        # Create some dummy data for testing
        N = 64
        NUMBER_OF_PULSES = 1000
        FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"

        # Load dataset
        pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)

        # Process data
        train_dataset, test_dataset = process_data(
            N, NUMBER_OF_PULSES, pulse_dataset, 0.8, 32)

        # Get one batch of the test dataset
        x, y = next(iter(test_dataset))

        # Compute the loss (should be zero)
        loss = custom_loss(x, y)

        # Print the value of the loss as a scalar
        print(f"Loss: {loss.numpy()}")

        # Check if the output is a tensor
        self.assertIsInstance(loss, tf.Tensor)

        # Check if the output tensor is a scalar (0-D tensor)
        self.assertEqual(loss.shape, ())

if __name__ == '__main__':
    unittest.main()