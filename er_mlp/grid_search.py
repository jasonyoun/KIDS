import er_mlp_max_margin
import itertools
import sys  
sys.stdout.flush()

print("running grid search...")
WORD_EMBEDDING = False
DATA_TYPE = 'freebase'
TRAINING_EPOCHS = 100 
BATCH_SIZE = 50000 
DISPLAY_STEP = 10

embedding_size_list = [50,60,70]
layer_size_list = [50,60,70]
learning_rate_list = [0.001,.01,0.1]
corrupt_size_list = [5, 10, 15]
lambda_list = [0.00001,0.0001, 0.001]
optimizer_list = [0,1]
optimal_list = None
optimal_accuracy = 0.
print('configuration initially set: ')
print('Word embedding: '+ str(WORD_EMBEDDING))
print('DATA_TYPE: '+ str(DATA_TYPE))
print('TRAINING_EPOCHS: '+ str(TRAINING_EPOCHS))
print('BATCH_SIZE: '+ str(BATCH_SIZE))
print('DISPLAY_STEP: '+ str(DISPLAY_STEP))
print('embedding_size_list: '+ str(embedding_size_list))
print('layer_size_list: '+ str(layer_size_list))
print('learning_rate_list: '+ str(learning_rate_list))
print('corrupt_size_list: '+ str(corrupt_size_list))
print('lambda_list: '+ str(lambda_list))
print('optimizer_list: '+ str(optimizer_list))
for x in itertools.product(embedding_size_list,layer_size_list,learning_rate_list,corrupt_size_list,lambda_list, optimizer_list):
	print('current configuration: ')
	print(x)
	EMBEDDING_SIZE = x[0]
	LAYER_SIZE = x[1]
	LEARNING_RATE = x[2]  
	CORRUPT_SIZE = x[3]
	LAMBDA = x[4]
	OPTIMIZER = x[5]
	accuracy = er_mlp_max_margin.run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, \
        LAYER_SIZE, TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER )
	if (accuracy > optimal_accuracy):
		optimal_accuracy = accuracy
		optimal_list = x
	print('optimal accuracy: ')
	print(optimal_accuracy)
	print('optimal configuration: ')
	print(optimal_list)

print('optimal configurations:')
print(optimal_list)
print('accuracy')
print(accuracy)