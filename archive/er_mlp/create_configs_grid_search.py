import itertools
import sys  
import pickle as pickle
sys.stdout.flush()

# embedding_size_list = [50,60,70]
# layer_size_list = [50,60,70]
# learning_rate_list = [0.001,.01,0.1]
# corrupt_size_list = [5, 10, 15]
# lambda_list = [0.00001,0.0001, 0.001]
# optimizer_list = [0,1]
# act_function = [0,1]
embedding_size_list = [60]
layer_size_list = [60]
learning_rate_list = [.01]
corrupt_size_list = [10]
lambda_list = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009, 0.001]
optimizer_list = [1]
act_function = [1]
add_layers = [0,1,2,3]
total_configs = list(itertools.product(embedding_size_list,layer_size_list,learning_rate_list,corrupt_size_list,lambda_list, optimizer_list, act_function, add_layers))

def save_object( obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(total_configs,'configs_'+str(len(total_configs))+'.txt')
# name = str(BATCH_SIZE)+str(DATA_TYPE)+str(WORD_EMBEDDING)+str(TRAINING_EPOCHS)+'.txt'
# file_name = open(name,'rb')
