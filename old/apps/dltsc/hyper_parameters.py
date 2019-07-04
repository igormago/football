import numpy as np
import json

class HyperParams:

    def __init__(self, argv):

        self.config = self.get_default_config(argv=argv)

        pass

    def get_default_config(self, argv):

        config = {}
        config['optim:num_epochs'] = 500000
        config['optim:progress_freqs'] = 100
        config['optim:batch_size'] = int(argv[6])
        config['optim:eta'] = float(argv[7])
        config['optim:max_grad_norm'] = 0.1
        config['optim:l2_lambda'] = 0
        config['optim:dropout_rate'] = float(argv[8])

        config['encoder_type'] = 'CNN'
        config['aggregation_type'] = 'flat'

        config['optim:loss_function'] = 'softmax_cross_entropy'

        config['rnn_encoder:layer_sizes'] = list(map(int, argv[11].split(',')))

        config['fcn:layer_sizes'] = []

        config['cnn_encoder:layer_sizes'] = list(map(int, argv[9].split(',')))
        config['cnn_encoder:kernel_sizes'] = list(map(int, argv[10].split(',')))
        config['cnn_encoder:strides'] = [1,1,1]

        # these settings have to be set for each class
        config['dataset:length'] = 96
        config['dataset:num_channels'] = 17
        config['dataset:num_classes'] = 3

        # parse argv and set the settings follow the command line options
        # to be implemented!!
        for arg in argv:
            if arg == '':
                pass

        return config

    def restore(file_path):
        return json.loads(file_path)

