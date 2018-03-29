import pickle as pickle
import er_mlp_max_margin
import itertools
import sys  
import os
directory = os.path.dirname(__file__)
sys.stdout.flush()
name = 'configs_40.txt'
name = os.path.join(directory, name)
file_name = open(name,'rb')
configs = pickle.load(file_name)
x= configs[int(sys.argv[1])]


print("running grid search...")
WORD_EMBEDDING = True
DATA_TYPE = 'freebase'
TRAINING_EPOCHS = 300 
BATCH_SIZE = 50000 
DISPLAY_STEP = 10
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
accuracy, auc = er_mlp_max_margin.run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, \
    LAYER_SIZE, TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS )

print('accuracy: ')
print(accuracy)
print('auc: ')
print(auc)


