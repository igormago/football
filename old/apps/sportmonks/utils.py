
def get_features(features, prefix, minute):

    if prefix == "trans":
        columns = ['possession_possession_home_' + str(minute)]
        prefix_trends = ['dif_counter_trends', 'div_counter_trends']
        prefix_cards = ['dif_counter_cards', 'div_counter_cards']

    for stat in features:

        if stat in ['yellow_cards', 'red_cards']:
            for t in prefix_cards:
                label = t + '_' + stat + '_' + str(minute)
                col = label
                columns.append(col)

        elif stat != 'possession':
            for t in prefix_trends:
                label = t + '_' + stat + '_' + str(minute)
                col = label
                columns.append(col)

    return columns


def split_train_test(data, features, target, odds=False):

    if odds:
        train_x = data[:25000][features]
        train_y = data[:25000][target]

        test_x = data[25001:][features]
        test_y = data[25001:][target]
    else:
        train_x = data[:40000][features]
        train_y = data[:40000][target]

        test_x = data[40001:][features]
        test_y = data[40001:][target]

    return train_x, train_y, test_x, test_y


def get_config(config):
    minute = int(config['minute'])
    try:
        optim_num_epochs = int(config['optim_num_epochs'])
    except KeyError:
        optim_num_epochs = 1000
    try:
        optim_batch_size = int(config['optim_batch_size'])
    except KeyError:
        optim_batch_size = 64
    try:
        optim_learning_rate = float(config['optim_learning_rate'])
    except KeyError:
        optim_learning_rate = 0.01
    try:
        cnn_encoder_layer_sizes = list(map(int, config['cnn_encoder_layer_sizes'].split(',')))
    except KeyError:
        cnn_encoder_layer_sizes = [128]
    try:
        cnn_encoder_kernel_sizes = list(map(int, config['cnn_encoder_kernel_sizes'].split(',')))
    except KeyError:
        cnn_encoder_kernel_sizes = [8]

    try:
        method = config['method']
    except KeyError:
        method = None

    try:
        dropout = float(config['optim:dropout_rate'])
    except KeyError:
        dropout = 0

    return cnn_encoder_kernel_sizes, cnn_encoder_layer_sizes, \
           minute, optim_batch_size, optim_learning_rate, optim_num_epochs, method, dropout