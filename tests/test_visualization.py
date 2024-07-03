import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import path_helper

from src.visualization import resultsGUI
from src.io import load_and_norm_data, process_data, process_data_tfrecord

import scienceplots


plt.style.use('science')
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = [11.69,8.27]
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.axisbelow'] = True


if __name__ == '__main__':
    # Define pulse database parameters
    N = 128
    Δt = 1 / N
    NUMBER_OF_PULSES = 1000
    FILE_PATH = f"./data/generated/N{N}/{NUMBER_OF_PULSES}_randomNormalizedPulses_N{N}.tfrecords"
    # Handle error if path does not exist
    try:
        with open(FILE_PATH) as f:
            pass
    except FileNotFoundError:
        print("File not found. Please generate the pulse database first.")
        exit()

    # Load and process the pulse database
    # pulse_dataset = load_and_norm_data(FILE_PATH, N, NUMBER_OF_PULSES)
    # train_dataset, test_dataset = process_data(N, NUMBER_OF_PULSES, pulse_dataset,
    #                                           0.8, 32)

    train_dataset, test_dataset = process_data_tfrecord(N, NUMBER_OF_PULSES, FILE_PATH, 0.1, 10, add_noise=False, noise_level=0.01, mask=True, mask_tolerance=1e-3)

    # Liberate memory from test_dataset
    del test_dataset

    # Load the trained model
    # model = keras.models.load_model(f"./trained_models/FCNN/MLP_test2.h5") #! Best MSE sweep result
    # model = keras.models.load_model(f"./trained_models/FCNN/bottleneck_MLP.h5") #! Bottleneck with MSE
    # model = keras.models.load_model(f"./trained_models/FCNN/MLP_test1.h5") #! Custom loss example
    # model = keras.models.load_model(f"./trained_models/CNN/CNN_test1.h5") #! Custom loss example with CNN + GPU!
    # model = keras.models.load_model(f"./trained_models/FCNN/bottleneck_MLP_custom_losstest1.h5") #! Custom loss example with CNN + GPU!
    # model = keras.models.load_model(f"./trained_models/CNN/CNN_test_N128_normTraces_total.tf") #! N=128
    # model = keras.models.load_model(f"./trained_models/CNN/CNN_test_N{N}_normTraces_combinedTraining_33.tf") #! N=128 with custom loss
    model = keras.models.load_model(f"./trained_models/CNN/CNN_test_N{N}_normTraces_combinedTraining_BIGDB.tf") #! N=128 with custom loss

    # # Lets see the output of the filters of the first and second conv layers
    # # Get the weights of the first layer
    # layer = model.get_layer(index=1)
    # weights = layer.get_weights()
    # filters = weights[0]
    # # Plot the filters
    # plt.figure()
    # for i in range(filters.shape[-1]):
    #     plt.subplot(4, 8, i+1)
    #     plt.imshow(filters[:,:,0,i], cmap='gray')
    #     plt.axis('off')
    #     # plot the color bar
    #     plt.colorbar()
    #     # plot the value of each weight in the filter
    #     for x in range(filters.shape[0]):
    #         for y in range(filters.shape[1]):
    #             plt.text(x, y, f"{filters[x,y,0,i]:.2f}", color='red', fontsize=6)

    #     # plot the value of the bias as title
    #     plt.title(f"Bias: {weights[1][i]:.2f}")



    # # plt.show()

    # # now lets see the the outputs of the first layer for an input of the model
    # # Get the input and output of the first layer
    # layer = model.get_layer(index=1)
    # intermediate_layer_model = keras.Model(inputs=model.input,
    #                                     outputs=layer.output)
    # # Get the first pulse of the training dataset
    # j = 0
    # for pulse, _ in train_dataset:
    #     pulse = pulse
    #     # Get the output of the first layer
    #     intermediate_output = intermediate_layer_model(pulse)
    #     # Plot the output of the first layer
    #     plt.figure()
    #     for i in range(intermediate_output.shape[-1]):
    #         plt.subplot(4, 8, i+1)
    #         plt.imshow(intermediate_output[0,:,:,i], cmap='nipy_spectral')
    #         plt.axis('off')


    #         # do the same for the second layer
    #     layer2 = model.get_layer(index=2)
    #     intermediate_layer_model2 = keras.Model(inputs=layer2.input,
    #                                         outputs=layer2.output)
    #     # Get the output of the first layer
    #     intermediate_output2 = intermediate_layer_model2(intermediate_output)
    #     # Plot the output of the first layer
    #     plt.figure()
    #     for i in range(intermediate_output2.shape[-1]):
    #         plt.subplot(4, 8, i+1)
    #         plt.imshow(intermediate_output2[0,:,:,i], cmap='nipy_spectral')
    #         plt.axis('off')

    #     # Layer 3
    #     layer3 = model.get_layer(index=3)
    #     intermediate_layer_model3 = keras.Model(inputs=layer3.input,
    #                                         outputs=layer3.output)

    #     intermediate_output3 = intermediate_layer_model3(intermediate_output2)

    #     plt.figure()
    #     for i in range(intermediate_output3.shape[-1]):
    #         plt.subplot(16, 8, i+1)
    #         plt.imshow(intermediate_output3[0,:,:,i], cmap='nipy_spectral')
    #         plt.axis('off')

    #     j += 1

    #     # Plt show if i is divisible by 5
    #     if j % 1 == 0:
    #         plt.show()
        


    # plt.show()

    # Prune the weights and biases in each layer and update them

    # for layer in model.layers:
    #     # Check that weights is not empty
    #     if layer.get_weights() != []:
    #         weights = layer.get_weights()[0]
    #         biases = layer.get_weights()[1]

    #         # Plot an historgram of the weights and biases
    #         fig, ax = plt.subplots(2, 1)
    #         ax[0].hist(weights.flatten(), bins=300)
    #         ax[0].set_title("Weights")
    #         ax[1].hist(biases.flatten(), bins=300)
    #         ax[1].set_title("Biases")

    #         # Set title with name of layer
    #         fig.suptitle(layer.name)

    #         # plt.show()


    #         # Prune all weights and biases if they are below a certain threshold
    #         weights[weights > 1e-5] = 0
    #         biases[biases > 1e-5] = 0

    #         # Update the weights and biases
    #         layer.set_weights([weights, biases])

    # Create the GUI
    results = resultsGUI(model, train_dataset, int(0.1 * NUMBER_OF_PULSES), N, Δt, norm_predictions=True, phase_blanking=True, phase_blanking_threshold=1e-5)
    results.plot()

    plt.show()