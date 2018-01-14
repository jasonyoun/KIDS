import numpy as np
import pickle as pickle
import pandas as pd
import sys
sys.path.insert(0, '../data')
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import utils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
# import matplotlib.pyplot as plt
import random
from tensorflow.python import debug as tf_debug

from data_processor import DataProcessor
from er_mlp import ERMLP
# from plotter import Plotter


WORD_EMBEDDING = True
DATA_TYPE = 'freebase'
EMBEDDING_SIZE = 60 # size of each embeddings
LAYER_SIZE = 60 # number of columns in the first layer
TRAINING_EPOCHS = 100 
BATCH_SIZE = 50000
LEARNING_RATE = 0.01  
DISPLAY_STEP = 1
CORRUPT_SIZE = 10
LAMBDA = 0.0001




# numerically represent the entities, predicates, and words
processor = DataProcessor()
print("machine translation...")
indexed_entities, num_entity_words, entity_dic,indexed_predicates, indexed_pred_word_embeddings, pred_dic,num_pred_words,num_entity_words = None,None,None,None,None,None,None,None
if WORD_EMBEDDING:
    indexed_entities, num_entity_words, entity_dic = processor.machine_translate_using_word('../data/raw/{}/entities.txt'.format(DATA_TYPE),EMBEDDING_SIZE, '../data/raw/{}/initEmbed.mat'.format(DATA_TYPE) )
    indexed_predicates, num_pred_words, pred_dic = processor.machine_translate_using_word('../data/raw/{}/relations.txt'.format(DATA_TYPE),EMBEDDING_SIZE)
else:
    entity_dic = processor.machine_translate('../data/raw/{}/entities.txt'.format(DATA_TYPE),EMBEDDING_SIZE)
    pred_dic = processor.machine_translate('../data/raw/{}/relations.txt'.format(DATA_TYPE),EMBEDDING_SIZE)

# load the data
train_df = processor.load('../data/raw/{}/train.txt'.format(DATA_TYPE))
test_df = processor.load('../data/raw/{}/test.txt'.format(DATA_TYPE))
dev_df = processor.load('../data/raw/{}/dev.txt'.format(DATA_TYPE))

# numerically represent the data 
print("Index:")
print(" - training complete")
indexed_data_training = processor.create_indexed_triplets_training(train_df.as_matrix(),entity_dic,pred_dic )
print(" - dev complete")
indexed_data_dev = processor.create_indexed_triplets_test(dev_df.as_matrix(),entity_dic,pred_dic )
print(" - test complete")
indexed_data_test = processor.create_indexed_triplets_test(test_df.as_matrix(),entity_dic,pred_dic )


NUM_ENTITIES = len(entity_dic)
NUM_PREDS = len(pred_dic)

er_mlp_params = {
    'word_embedding': WORD_EMBEDDING,
    'embedding_size': EMBEDDING_SIZE,
    'layer_size': LAYER_SIZE,
    'corrupt_size': CORRUPT_SIZE,
    'lambda': LAMBDA,
    'num_entities':NUM_ENTITIES,
    'num_preds':NUM_PREDS,
    'num_entity_words':num_entity_words,
    'num_pred_words':num_pred_words,
    'learning_rate':LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'indexed_entities':indexed_entities,
    'indexed_predicates': indexed_predicates,
}

plotter_params = {
    'batch_size': BATCH_SIZE,
    'training_epochs': TRAINING_EPOCHS,
    'data_type': DATA_TYPE,
    'num_entities':NUM_ENTITIES,
    'num_preds':NUM_PREDS,
    'pred_dic':pred_dic
}

er_mlp = ERMLP(er_mlp_params)

triplets, weights, biases, constants, y = er_mlp.create_tensor_terms()

# The training triplets: subject, predicate, object, corrupted_entity
training_triplets = tf.placeholder(tf.int32, shape=(None, 4))

# A boolean to determine if we want to corrupt the head or tail
flip_placeholder = tf.placeholder(tf.bool)

training_predictions = er_mlp.inference_for_max_margin_training(training_triplets, weights, biases, constants, flip_placeholder)

print('network for predictions')
predictions = er_mlp.inference(triplets, weights, biases, constants)

print('calculate cost')
cost = er_mlp.loss(training_predictions)

print('optimizer')
optimizer = er_mlp.train(cost)

# plotter = Plotter(plotter_params)




print("initialize tensor variables")
init_all = tf.global_variables_initializer()

print("begin tensor seesion")
sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(init_all)


data_train = indexed_data_training[:,:3]

iter_list = []
cost_list = []
iteration = 0
current_cost = 0.
print("Begin training...")
for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0.
    total_batch = int(data_train.shape[0] / BATCH_SIZE)
    for i in range(total_batch):
        batch_xs = er_mlp.get_training_batch_with_corrupted(data_train)
        flip = bool(random.getrandbits(1))
        _, current_cost= sess.run([optimizer, cost], feed_dict={training_triplets: batch_xs, flip_placeholder: flip})
        avg_cost +=current_cost/total_batch
        print(current_cost)
        cost_list.append(current_cost)
        iter_list.append(iteration)
        iteration+=1
    # Display progress
    if epoch % DISPLAY_STEP == 0:
        print ("Epoch: %03d/%03d cost: %.9f - current_cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost,current_cost ))

# use the dev set to compute the best threshold for classification
print("determine threshold for classification")
data_dev = indexed_data_dev[:,:3]
predicates_dev = indexed_data_dev[:,1]
labels_dev = indexed_data_dev[:,3]
labels_dev = labels_dev.reshape((np.shape(labels_dev)[0],1))
predictions_dev = sess.run(predictions, feed_dict={triplets: data_dev, y: labels_dev})
threshold = er_mlp.compute_threshold(predictions_dev,labels_dev,predicates_dev)


# Test the model by classifying each sample using the threshold determined by running the model
# over the dev set
print("test model")
data_test = indexed_data_test[:,:3]
labels_test = indexed_data_test[:,3]
labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
predicates_test = indexed_data_test[:,1]
predictions_list = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})
classifications = er_mlp.classify(predictions_list,threshold, predicates_test)
accuracy = sum(1 for x,y in zip(labels_test,classifications) if x == y) / len(labels_test)
print("overall accuracy:")
print(accuracy)

# find the accuracy for the baseline, which is the most often occuring class
a = np.empty(np.shape(classifications))
a.fill(-1)
accuracy_b = sum(1 for x,y in zip(labels_test,a) if x == y) / len(a)
print("baseline accuracy:")
print(accuracy_b)

# plotter.plot_roc(labels_test,predictions_list,predicates_test)
# plotter.plot_pr(labels_test,predictions_list,predicates_test)
# plotter.plot_cost(iter_list,cost_list)

def save_object( obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_data = {
    'prediction_list': predictions_list,
    'labels_test':labels_test,
    'batch_size': BATCH_SIZE,
    'training_epochs': TRAINING_EPOCHS,
    'data_type': DATA_TYPE,
    'num_entities':NUM_ENTITIES,
    'num_preds':NUM_PREDS,
    'pred_dic':pred_dic,
    'predicates': predicates_test

}
save_object(save_data,str(BATCH_SIZE)+str(DATA_TYPE)+str(WORD_EMBEDDING)+str(TRAINING_EPOCHS)+'.txt')
name = str(BATCH_SIZE)+str(DATA_TYPE)+str(WORD_EMBEDDING)+str(TRAINING_EPOCHS)+'.txt'
file_name = open(name,'rb')







