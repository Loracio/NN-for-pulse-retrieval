# First steps for approaching the problem

## Pulse database

The task that we want the Neural Network to permorm is to *inverse map* the $N\times N$ real values that represent the SHG-FROG trace to the $2N$ real numbers that represent the real and imaginary parts of the electric field values.

<p align="center">
  <img src="./figs/readme/NN.png" alt="NN scheme for solving the retrieval problem." width="550"/><br>
  <em>Figure 1: Scheme of the input and output layers of a Neural Network that solves the retrieval problem.</em>
</p>

Therefore, we will need to generate a database of simulated pulses containing both their SHG-FROG trace and the real and imaginary parts of the pulse electric field. An algorithm to perform such a task can be found at [this link](https://github.com/Loracio/ultrafast-pulse-retrieval/blob/main/src/pulse.hpp#L446) to the [`ultrafast-pulse-retrieval`](https://github.com/Loracio/ultrafast-pulse-retrieval) project, and can be used to create a database as required in this problem, as shown in the file [`testDataBaseGeneration.cpp`](https://github.com/Loracio/ultrafast-pulse-retrieval/blob/main/tests/testDataBaseGeneration.cpp).

The generated databases are .csv files containing the following structure:

|TBP | $E_{Re \ 0}$ | $\dots$ | $E_{Re \ N-1}$ | $E_{Im \ 0}$ | $\dots$ | $E_{Im \ N-1}$ | $T_{00}$ | $\dots$ | $T_{N-1 \ N-1}$

where:

- TBP: Time bandwidth product
- $E_{Re}$: Pulse real part (N columns)
- $E_{Im}$: Pulse imaginary part (N columns)
- $T_{mn}$: SHG-FROG trace of the pulse (NxN columns)

**Tensorflow** will be the library used in this project for the creation and training of Neural Networks, so the database has to be read and processed to be compatible with tensorflow data types.

A good practice is to normalize the input and the output of the Neural Network training and validation data. In this case, each trace is normalized by dividing by its maximum value, and the real and imaginary parts of each pulse are divided by the maximum amplitude (polar representation of complex numbers).

Database input/output handling functions are implemented inside the [`src/io`](/src/io/) folder, and will read and format the data into training and validation sets to train the Neural Network.

Now that we've got the data for training the Neural Network, we have to actually design and train it to solve our problem. We will use a *construction* approach, starting with the simplest possible model and gradually adding features that make it more complex and predictably better at solving the problem.

For a better handling and a faster training we will be using pulses with **N=64** for now on. This places limitations on the time-bandwidth product of the pulses we will be able to simulate, and therefore, on the ability of the network to produce pulses of a higher TBP.

## Multi Layer Perceptron as a first approach

The simplest architecture that can be used to solve the problem is a Multi Layer Perceptron (MLP), which is based on the basic unit of a neural network: the perceptron.

The architecture of this network will consist of an input layer with the NxN values of the SHG-FROG trace, one or more hidden layers of a variable number of neurons and an output with the 2N values of the electric field (real and imaginary parts).

There are several choices of hyperparameters that must be made to build the neural network:

- **Epochs**: number of epochs in which the network will be trained. To avoid overfitting, it is desirable to define an "early stopping" so that the training is stopped before the total number of epochs is reached if there is no improvement on the validation set.
- **Batch size**: number of training examples used in one iteration of the gradient descent. When the batch size is small, each update to the model parameters is based on fewer examples from the training set. This leads to more noise in the update process, which can help the model escape from local minima but also makes the training process less stable. On the other hand, when the batch size is large, each update is based on more examples, which makes the update process less noisy and more stable, but also more likely to get stuck in local minima. This is closer to the behavior of "batch" gradient descent, where each update is based on the entire training set.
- **Optimizer**: the optimizer in a neural network determines how the model updates its parameters in response to the calculated error gradient. The error gradient, which is computed using backpropagation, indicates the direction in which the model parameters need to be adjusted to minimize the loss function.
- **Learning rate**: the size of the steps taken to reach the minimum of the loss function. If the learning rate is too large, it may overshoot the optimal point. If it's too small, training will take too long.
- **Loss**: The function that the model tries to minimize during training. It's a measure of how far the model's predictions are from the true values.
- **Training and validation metrics**: metrics used to measure the performance of the model on the training and validation datasets.
- **Number of hidden layers**: number of layers in the neural network that are neither the input nor the output layer.
- **Number of neurons per hidden layer**: number of neurons in each hidden layer.
- **Activation function**: the function used to transform the input signal of a neuron in the network to its output signal.
- **Initializer**: the initializer of weights and biases in a neural network is a method or function that sets the initial values of the model's parameters (weights and biases) before training starts. The choice of initialization can significantly affect the speed of convergence and the final performance of the model.
- **Dropout**: regularization technique where randomly selected neurons are ignored during training. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass. It helps prevent overfitting.

Deciding which of these parameters to use often comes down to a process of trial and error, although it is possible to choose them with "a bit of common sense".

TBC... Check out [`/notebooks/first_steps.ipynb`](/notebooks/first_steps.ipynb)
