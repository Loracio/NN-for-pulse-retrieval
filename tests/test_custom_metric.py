import unittest
import tensorflow as tf

import path_helper

from src.models import trace_MSE, trace_loss
from src.io import load_and_norm_data, process_data

class TestCustomMetric(unittest.TestCase):
    def test_custom_metric(self):
        N = 64
        NUMBER_OF_PULSES = 1000
        FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomPulses_N{N}.csv"

        # Load dataset
        pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)

        # Process data
        train_dataset, test_dataset = process_data(
            N, NUMBER_OF_PULSES, pulse_dataset, 0.8, 32)

        train_acc_metric = trace_MSE(N, 1/N)
        loss_fn = trace_loss(N, 1/N)

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss = loss_fn(x_batch_train, y_batch_train)
            train_acc_metric.update_state(x_batch_train, y_batch_train)
            # Print the value of the loss
            print(f"Loss: {loss}")

        train_acc = train_acc_metric.result()
        
        print(f"Metric: {train_acc}")
        train_acc_metric.reset_states()

        # Check if the output is a tensor
        self.assertIsInstance(train_acc, tf.Tensor)

        # Check if the output tensor is a scalar (0-D tensor)
        self.assertEqual(train_acc.shape, ())

if __name__ == '__main__':
    unittest.main()