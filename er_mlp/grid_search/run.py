import pickle as pickle
import er_mlp_cross_entropy
import itertools
import sys  
import os
import configparser
import json
directory = os.path.dirname(__file__)
sys.stdout.flush()
config = configparser.ConfigParser()
print(sys.argv[1]+'/er_mlp.ini')
config.read(sys.argv[1]+'/er_mlp.ini')


print("running grid search...")
print(config['DEFAULT']['EMBEDDING_SIZE'])
embedding_size_list = json.loads(config['DEFAULT']['EMBEDDING_SIZE'])
layer_size_list = json.loads(config['DEFAULT']['LAYER_SIZE'])
learning_rate_list =  json.loads(config['DEFAULT']['LEARNING_RATE'])
corrupt_size_list =  json.loads(config['DEFAULT']['CORRUPT_SIZE'])
lambda_list = json.loads(config['DEFAULT']['LAMBDA'])
optimizer_list =  json.loads(config['DEFAULT']['OPTIMIZER'])
act_function =  json.loads(config['DEFAULT']['ACT_FUNCTION'])
add_layers =  json.loads(config['DEFAULT']['ADD_LAYERS'])
drop_out_precent =  json.loads(config['DEFAULT']['ADD_LAYERS'])
total_configs = list(itertools.product(embedding_size_list,layer_size_list,learning_rate_list,corrupt_size_list,lambda_list, optimizer_list, act_function, add_layers, drop_out_precent))

x= total_configs[int(sys.argv[2])]
DATA_PATH = sys.argv[1]
WORD_EMBEDDING = False
DATA_TYPE = 'ecoli'
TRAINING_EPOCHS = 100
BATCH_SIZE = 500
DISPLAY_STEP = 1
EMBEDDING_SIZE = x[0]
LAYER_SIZE = x[1]
LEARNING_RATE = x[2]  
CORRUPT_SIZE = x[3]
LAMBDA = x[4]
OPTIMIZER = x[5]
ACT_FUNCTION = x[6]
ADD_LAYERS = x[7]
DROP_OUT_PERCENT = x[8]


er_mlp_cross_entropy.run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, \
    LAYER_SIZE, TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS, DROP_OUT_PERCENT,DATA_PATH )

