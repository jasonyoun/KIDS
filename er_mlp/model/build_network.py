import pickle as pickle
import itertools
import sys  
import os
import configparser
directory = os.path.dirname(__file__)
abs_path_er_mlp= os.path.join(directory, '../er_mlp_imp')
sys.path.insert(0, abs_path_er_mlp)
import er_mlp_cross_entropy
import er_mlp_max_margin
sys.stdout.flush()
config = configparser.ConfigParser()
configuration = sys.argv[1]+'.ini'
print('./'+configuration)
config.read('./'+configuration)


print('configuration: ')
WORD_EMBEDDING = config.getboolean('DEFAULT','WORD_EMBEDDING')
DATA_TYPE = config['DEFAULT']['DATA_TYPE']
TRAINING_EPOCHS = config.getint('DEFAULT','TRAINING_EPOCHS')
BATCH_SIZE = config.getint('DEFAULT','BATCH_SIZE')
DISPLAY_STEP =  config.getint('DEFAULT','DISPLAY_STEP')
EMBEDDING_SIZE = config.getint('DEFAULT','EMBEDDING_SIZE')
LAYER_SIZE = config.getint('DEFAULT','LAYER_SIZE')
LEARNING_RATE = config.getfloat('DEFAULT','LEARNING_RATE')
CORRUPT_SIZE = config.getint('DEFAULT','CORRUPT_SIZE')
LAMBDA = config.getfloat('DEFAULT','LAMBDA')
OPTIMIZER = config.getint('DEFAULT','OPTIMIZER')
ACT_FUNCTION = config.getint('DEFAULT','ACT_FUNCTION')
ADD_LAYERS = config.getint('DEFAULT','ADD_LAYERS')
DROP_OUT_PERCENT = config.getfloat('DEFAULT','ADD_LAYERS')
DATA_PATH=config['DEFAULT']['DATA_PATH']
SAVE_MODEL=True
MODEL_SAVE_DIRECTORY=config['DEFAULT']['MODEL_SAVE_DIRECTORY']
MAX_MARGIN_TRAINING = config.getboolean('DEFAULT','MAX_MARGIN_TRAINING')
USE_NEG = config.getboolean('DEFAULT','USE_NEG')
USE_RANGE = config.getboolean('DEFAULT','USE_RANGE')

if MAX_MARGIN_TRAINING:
    er_mlp_max_margin.run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, \
        LAYER_SIZE, TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS, DROP_OUT_PERCENT,DATA_PATH, SAVE_MODEL , MODEL_SAVE_DIRECTORY,USE_NEG,USE_RANGE)
else:
    er_mlp_cross_entropy.run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, \
        LAYER_SIZE, TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS, DROP_OUT_PERCENT,DATA_PATH, SAVE_MODEL , MODEL_SAVE_DIRECTORY)

