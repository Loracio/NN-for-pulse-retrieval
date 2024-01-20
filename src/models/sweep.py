import tensorflow as tf

# TODO: check and fix this function
def sweep_MLP(sweep_config, FILE_PATH, N, NUMBER_OF_PULSES):

    # Initialize Weights & Biases
    run = wandb.init(config=sweep_config)

    # Specify the other hyperparameters to the configuration, if any
    wandb.config.log_step = LOG_STEP
    wandb.config.val_log_step = VAL_LOG_STEP
    # The architecture name is given by the number of hidden layers, neurons and activation function. Also the optimizer, learning rate and batch size
    # wandb.config.architecture_name = f"L={wandb.config.hidden_layer_number}_N={wandb.config.hidden_layer_neurons}_A={wandb.config.activation}_OP={wandb.config.optimizer}_LR={wandb.config.learning_rate}_BATCHSIZE={wandb.config.batch_size}"
    # wandb.config.dataset_name = f"{NUMBER_OF_PULSES}_randomPulses_N{N}"

    # Load dataset
    pulse_dataset = load_data(FILE_PATH, N, NUMBER_OF_PULSES)

    def select_yz(x, y, z):
        return (y, z)

    pulse_dataset = pulse_dataset.map(select_yz)

    # Split the dataset into train and test, shuffle and batch the train dataset
    train_dataset = pulse_dataset.take(int(0.75 * NUMBER_OF_PULSES)).shuffle(buffer_size=NUMBER_OF_PULSES).batch(wandb.config.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = pulse_dataset.skip(int(0.75 * NUMBER_OF_PULSES)).batch(wandb.config.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # initialize model
    model = make_model(number_hidden_layers=wandb.config.hidden_layer_number, 
                       hidden_layer_neurons=wandb.config.hidden_layer_neurons, 
                       activation=wandb.config.activation)

    train(train_dataset,
          val_dataset, 
          model,
          keras.optimizers.get(wandb.config.optimizer),
          keras.losses.get(wandb.config.loss),
          tf.keras.metrics.MeanSquaredError(), # In this case the test accuracy is the same as the test loss
          tf.keras.metrics.MeanSquaredError(), # In this case the train accuracy is the same as the test loss
          epochs=wandb.config.epochs, 
          log_step=wandb.config.log_step, 
          val_log_step=wandb.config.val_log_step)