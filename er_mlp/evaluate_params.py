import pickle as pickle
import er_mlp_max_margin
import itertools
import sys  
import os
from data_processor import DataProcessor
from orchestrator import Orchestrator
directory = os.path.dirname(__file__)
if directory != '':
    directory = directory+'/'
sys.stdout.flush()
name = 'configs_40.txt'
name = os.path.join(directory, name)
file_name = open(name,'rb')
configs = pickle.load(file_name)
x= configs[int(sys.argv[1])]


print("running grid search...")
WORD_EMBEDDING = False
DATA_TYPE = 'ecoli'
TRAINING_EPOCHS = 20 
BATCH_SIZE = 1000 
DISPLAY_STEP = 10
MODEL_PATH = './'
print('current configuration: ')
print(x)
EMBEDDING_SIZE = x[0]
LAYER_SIZE = x[1]
LEARNING_RATE = x[2]  
CORRUPT_SIZE = x[3]
LAMBDA = x[4]
OPTIMIZER = x[5]
ACT_FUNCTION = x[6]
ADD_LAYERS = x[7]
DROP_OUT_PERCENT = 0.5


params = {
    'word_embedding': WORD_EMBEDDING,
    'embedding_size': EMBEDDING_SIZE,
    'layer_size': LAYER_SIZE,
    'corrupt_size': CORRUPT_SIZE,
    'lambda': LAMBDA,
    'learning_rate':LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'add_layers': ADD_LAYERS,
    'data_type': DATA_TYPE,
    'training_epochs': TRAINING_EPOCHS,
    'display_step': DISPLAY_STEP,
    'optimizer':OPTIMIZER,
    'act_function':ACT_FUNCTION,
    'add_layers': ADD_LAYERS,
    'drop_out_percent':DROP_OUT_PERCENT
}   
model_id='1519444328362'

orch = Orchestrator(params=params)
print(orch.get_model_id())
orch.load_data()
orch.create_network()
orch.train_network(dataset='train')
accuracy,auc = orch.eval_network(dataset='test')

# processor = DataProcessor()
# predict_df = processor.load(directory+'../data/raw/{}/test.txt'.format(DATA_TYPE))
# print(orch.predict(predict_df.as_matrix()))
orch.delete_model()


print('accuracy: ')
print(accuracy)
print('auc: ')
print(auc)