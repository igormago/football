import tensorflow as tf
from prediction_model import PredictionModel


class CNNAndRNN(PredictionModel):

    def __init__(self, config):
        PredictionModel.__init__(self, config)

        self.name = 'SupervisedAggregationModel'

    def create_embedding(self):

        print(self.config)

        with tf.name_scope("CNNAndRNNEncoder"):
            feature_map = self.X_batch

            for num_filters, kernel_length, stride in zip(self.config['cnn_encoder:layer_sizes'],
                                                          self.config['cnn_encoder:kernel_sizes'],
                                                          self.config['cnn_encoder:strides']):

            #for num_filters in self.config['cnn_encoder:layer_sizes']:

                #len = feature_map.shape[1].value
                #kernel_length = len // 10 + 1
                #stride = kernel_length // 5 + 1

                #kernel_length = 1
                #stride = 1

                feature_map = tf.layers.conv1d(inputs=feature_map, filters=num_filters, padding='VALID',
                                               kernel_size=kernel_length,
                                               strides=stride)
                # add batch normalization
                feature_map = tf.layers.batch_normalization(feature_map, training=self.is_training)
                # add the feature map
                feature_map = tf.nn.relu(feature_map)

                print('Add CNN layer', feature_map, ' kernel_length', kernel_length, 'stride', stride)

            # pass the last feaure map through drop out to create the embedding
            self.h = feature_map

            print('CNN Encoder', self.h)

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
                inputs=tf.unstack(self.h, axis=1),
                dtype=tf.float32)
            # stack the outputs list into a tensor
            self.h = tf.stack(outputs)
            # swap the axes in order to have (batch size, timernn_encoder:layer_sizes, cells) as the output
            self.h = tf.transpose(self.h, [1, 0, 2])

            # apply batch normalization to the RNN activations
            self.h = tf.layers.batch_normalization(self.h,
                                                   training=self.is_training,
                                                   name='RNNActivationBatchNorm')
            # apply dropout for regularization purpose
            self.h = tf.layers.dropout(self.h, rate=self.config['optim:dropout_rate'],
                                       training=self.is_training, name='RNNActivationDropOut')
            print('CNNAndRNN Encoder', self.h)


    def create_target_estimation(self):

        with tf.name_scope("Target"):

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

            self.Y_hat = tf.reduce_max(self.Y_hat, axis=1, name="Y_hat_mean")

