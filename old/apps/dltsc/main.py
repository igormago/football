from optimizer import Optimizer
from hyper_parameters import HyperParams
from dataset import Dataset
from prediction_model import PredictionModel
from CNNAndRNN import CNNAndRNN
import sys

# get the configurations
hp = HyperParams(argv=sys.argv)
config = hp.get_default_config(sys.argv)

# load the dataset
dataset_folder = sys.argv[1]
encoder = sys.argv[2]
aggregation = sys.argv[3]
model_type = sys.argv[4]
crop = sys.argv[5]
batch_size = sys.argv[6]
learning_rate = sys.argv[7]
dropout = sys.argv[8]
cnn_layers = sys.argv[9]
kernels = sys.argv[10]
if sys.argv[11]:
    rnn_layers = sys.argv[11]

name_folder = dataset_folder.split('/')[-2]

title = name_folder + "_" + str(crop) + "_"
for i in range(2, len(sys.argv)):
    title = title + sys.argv[i] + "_"
config['title'] = title

crop = int(crop) + 1

dataset = Dataset()
dataset.load_multivariate(dataset_folder, crop)
#dataset.load_ucr_univariate_data(dataset_folder=dataset_folder)
config['dataset:length'] = dataset.series_length
config['dataset:num_channels'] = dataset.num_channels
config['dataset:num_classes'] = dataset.num_classes

# create the model
model = None

config['encoder_type'] = sys.argv[2]
config['aggregation_type'] = sys.argv[3]

# create the prediction model

if model_type == 'standard':
    model = PredictionModel(config)
elif model_type == 'supervised':
    model = CNNAndRNN(config)

model.create_prediction_model()

print("This model has", model.num_model_parameters(), "parameters")


opt = Optimizer(config=config, dataset=dataset, model=model)
opt.optimize()
