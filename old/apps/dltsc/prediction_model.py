import tensorflow as tf

class PredictionModel:

    def __init__(self, config):

        with tf.name_scope("PredictionModel"):
            self.config = config
            self.name = 'PredictionModel'

            # the time series and the targets of the batch
            self.X_batch = tf.placeholder(shape=(self.config['optim:batch_size'],
                                                 self.config['dataset:length'],
                                                 self.config['dataset:num_channels']),
                                          dtype=tf.float32,
                                          name="X_batch")
            # the target classes
            self.Y_batch = tf.placeholder(shape=(self.config['optim:batch_size'], self.config['dataset:num_classes']),
                                          dtype=tf.float32,
                                          name="Y_batch")

            # a boolean that indicates whether the predictions called are in the training
            # or inference phase, useful for batch normalization
            self.is_training = tf.placeholder(tf.bool)

            # the encoded representation
            self.h = None

            # the prediction model output, the estimated targets
            self.Y_hat = None

            # the update rule for optimizing the model
            self.update_rule = None



    # this is the embedding layer of Y_batch, i.e. the last convolutional feature map
    # or the activations of the last LSTM cells, etc ...
    def create_embedding(self):

        if self.config['encoder_type'] == 'CNN':
            with tf.name_scope("CNNEncoder"):

                print(self.config)
                feature_map = self.X_batch

                for num_filters, kernel_length, stride in zip(self.config['cnn_encoder:layer_sizes'],
                                                              self.config['cnn_encoder:kernel_sizes'],
                                                              self.config['cnn_encoder:strides']):
                    num_filters = int(num_filters)
                    kernel_length = int(kernel_length)

                    feature_map = tf.layers.conv1d(inputs=feature_map, filters=num_filters, padding='VALID',
                                                   kernel_size=kernel_length,
                                                   strides=stride)
                    # add batch normalization
                    feature_map = tf.layers.batch_normalization(feature_map, training=self.is_training)
                    # add the feature map
                    feature_map = tf.nn.relu(feature_map)

                    print('Add CNN layer', feature_map, ' kernel_length', kernel_length, 'stride', stride)

                print(self.config['optim:dropout_rate'])
                # pass the last feaure map through drop out to create the embedding
                self.h = tf.layers.dropout(feature_map, rate=self.config['optim:dropout_rate'],
                                           training=self.is_training, name='CNNActivation')

                print('CNN Encoder', self.h)

        elif self.config['encoder_type'] == 'RNN':

            print('Creating RNN encoder')
            cells_fwd_list = []
            for num_cells in self.config['rnn_encoder:layer_sizes']:
                print("Forward RNN layer with", num_cells, "cells")
                cells_fwd_list.append(tf.nn.rnn_cell.LSTMCell(num_units=num_cells, activation=tf.nn.tanh))
                self.cells_fwd = tf.nn.rnn_cell.MultiRNNCell(cells_fwd_list, state_is_tuple=True)

            cells_bwd_list = []
            for num_cells in self.config['rnn_encoder:layer_sizes']:
                print("Backward RNN layer with", num_cells, "cells")
                cells_bwd_list.append(tf.nn.rnn_cell.LSTMCell(num_units=num_cells, activation=tf.nn.tanh))
                self.cells_bwd = tf.nn.rnn_cell.MultiRNNCell(cells_bwd_list, state_is_tuple=True)

            # create a dynamic rnn with a specified sequence length
            # which outputs the activations and states of the last LSTM's layer for each time index
            (outputs, state_fw, state_bw) = tf.nn.static_bidirectional_rnn(
                cell_fw=self.cells_fwd,
                cell_bw=self.cells_bwd,
                inputs=tf.unstack(self.X_batch, axis=1),
                dtype=tf.float32)
            # stack the outputs list into a tensor
            self.h = tf.stack(outputs)
            # swap the axes in order to have (batch size, time, cells) as the output
            self.h = tf.transpose(self.h, [1, 0, 2])
            # apply batch normalization to the RNN activations
            self.h = tf.layers.batch_normalization(self.h,
                                                   training=self.is_training,
                                                   name='RNNActivationBatchNorm')
            # apply dropout for regularization purpose
            self.h = tf.layers.dropout(self.h, rate=self.config['optim:dropout_rate'],
                                       training=self.is_training, name='RNNActivationDropOut')

            print('RNN Encoder', self.h)

    # this creates the estimated targets Y_hat from the embedding layer
    def create_target_estimation(self):

        with tf.name_scope("Target"):

            # conduct the aggregation of the layers
            if self.config['aggregation_type'] == 'flat':
                # flatten the embedding layer
                self.h = tf.reshape(tensor=self.h,
                                    shape=(self.config['optim:batch_size'], -1),
                                    name='FlattenedEmbedding')
                print('Flattened layer', self.h)

            elif self.config['aggregation_type'] == 'mean':
                # flatten the embedding layer
                self.h = tf.reduce_mean(input_tensor=self.h, axis=1, name="AggregatedEmbedding")
                print('Mean layer', self.h)

            # define the fully connected layer
            self.Y_hat = self.h
            for idx, num_neurons in enumerate(self.config['fcn:layer_sizes']):
                # the dense layer of the fully connected architecture
                self.Y_hat = tf.layers.dense(inputs=self.Y_hat,
                                             units=num_neurons,
                                             activation=tf.nn.relu,
                                             name='YHatLayer' + str(idx))
                # batch norm layer
                self.Y_hat = tf.layers.batch_normalization(self.Y_hat,
                                                           training=self.is_training,
                                                           name='YHatBatchNorm' + str(idx))

                print('NN layer', idx, self.Y_hat, 'num_neurons', num_neurons)

            # the output layer
            self.Y_hat = tf.layers.dense(inputs=self.Y_hat,
                                         units=self.config['dataset:num_classes'],
                                         name='YHat')




    # define the modelm by first creating the latent embedding and then the target estimation
    def create_prediction_model(self):
        self.create_embedding()
        self.create_target_estimation()

        tf.summary.FileWriter('/home/igorcosta/tensorlogs', self.model)

    # count the number of parameters in the model
    def num_model_parameters(self):

        total_parameters = 0

        for variable in tf.trainable_variables():
            shape = variable.get_shape()

            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value

            total_parameters += variable_parameters

        return total_parameters
