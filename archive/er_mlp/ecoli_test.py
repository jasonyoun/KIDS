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
import matplotlib.pyplot as plt
import random
from tensorflow.python import debug as tf_debug

from data_processor import DataProcessor
from er_mlp import ERMLP
from plotter import Plotter

OVER_SAMPLE = True
WEIGHTED_LOSS = False
WORD_EMBEDDING = False
DATA_TYPE = 'ecoli'
EMBEDDING_SIZE = 60 # size of each embeddings
LAYER_SIZE = 60 # number of columns in the first layer
TRAINING_EPOCHS = 350 
BATCH_SIZE = 5000
LEARNING_RATE = 0.001  
DISPLAY_STEP = 1
CORRUPT_SIZE = 10
LAMBDA = 0.0001
TEST_SIZE = 0.3

processor = DataProcessor()

# load the data
df = processor.load('../data/raw/{}/hiTRN_KG_without_multi-gene-TFs.txt'.format(DATA_TYPE))
#df = processor.load('../data/raw/{}/test.txt'.format(DATA_TYPE))

entity_dic = processor.machine_translate('../data/raw/{}/entities.txt'.format(DATA_TYPE),EMBEDDING_SIZE)
pred_dic = processor.machine_translate('../data/raw/{}/relations.txt'.format(DATA_TYPE),EMBEDDING_SIZE)

indexed_data = processor.create_indexed_triplets_test(df.as_matrix(),entity_dic,pred_dic )
data_train, data_test, labels_train, labels_test = train_test_split(indexed_data[:,:3],indexed_data[:,3], test_size=TEST_SIZE)

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
    'num_entity_words':None,
    'num_pred_words':None,
    'learning_rate':LEARNING_RATE,
    'batch_size': BATCH_SIZE,
}

plotter_params = {
    'batch_size': BATCH_SIZE,
    'training_epochs': TRAINING_EPOCHS,
    'data_type': DATA_TYPE,
    'num_entities':NUM_ENTITIES,
    'num_preds':NUM_PREDS,
    'pred_dic':pred_dic
}

plotter = Plotter(plotter_params)
er_mlp = ERMLP(er_mlp_params)

triplets, weights, biases, constants, y = er_mlp.create_tensor_terms()

print('network for predictions')
predictions = er_mlp.inference(triplets, weights, biases, constants)

cost = None
if WEIGHTED_LOSS:
    cost = er_mlp.loss_weighted_cross_entropy(predictions,y)
else:
    cost = er_mlp.loss_cross_entropy(predictions,y)
print('optimizer')
optimizer = er_mlp.train(cost)


print("initialize tensor variables")
init_all = tf.global_variables_initializer()

print("begin tensor seesion")
sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(init_all)

if OVER_SAMPLE:
    s = data_train[:,0]
    s = s.reshape((np.shape(s)[0],1))
    p = data_train[:,1]
    p = p.reshape((np.shape(p)[0],1))
    o = data_train[:,2]
    o = o.reshape((np.shape(o)[0],1))
    t = labels_train.reshape((np.shape(labels_train)[0],1))
    combined_data = np.concatenate((s,p,o,t), axis=1)
    positive_samples = combined_data[combined_data[:,3] == 1]
    negative_samples = combined_data[combined_data[:,3] == 0]
    allIdx = np.array(range(0,np.shape(positive_samples)[0]))
    idx = np.random.choice(allIdx,size=np.shape(negative_samples)[0],replace=True)
    os_positive_samples = positive_samples[idx]
    combined_data = np.concatenate((os_positive_samples,negative_samples), axis=0)
    data_train = combined_data[:,:3]
    labels_train = combined_data[:,3]
    print(np.shape(positive_samples))
    print(np.shape(os_positive_samples))
    print(np.shape(negative_samples))
    print(np.shape(combined_data))

iter_list = []
cost_list = []
iteration = 0
print("Begin training...")
for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0.
    total_batch = int(data_train.shape[0] / BATCH_SIZE)
    for i in range(total_batch):
        randidx = np.random.randint(int(data_train.shape[0]), size = BATCH_SIZE)

        batch_xs = data_train[randidx, :]
        batch_ys = labels_train[randidx]
        batch_ys = batch_ys.reshape((np.shape(batch_ys)[0],1))

        _, current_cost= sess.run([optimizer, cost], feed_dict={triplets: batch_xs, y: batch_ys})
        avg_cost +=current_cost/total_batch
        print(current_cost)
        cost_list.append(current_cost)
        iter_list.append(iteration)
        iteration+=1
    # Display progress
    if epoch % DISPLAY_STEP == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost))

labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
predicates_test = data_test[:,1]
predictions_list = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})
probabilities = tf.sigmoid(predictions_list)
probabilities_np = probabilities.eval(session=sess)
plotter.plot_roc(labels_test,probabilities_np,predicates_test)
plotter.plot_pr(labels_test,probabilities_np,predicates_test)
plotter.plot_cost(iter_list,cost_list)

def save_object( obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_data = {
    'prediction_list': probabilities_np,
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
wv = pickle.load(file_name)
wv['prediction_list']
wv['labels_test']
wv['batch_size']
wv['training_epochs']
wv['data_type']
wv['num_entities']
wv['num_preds']
wv['pred_dic']
wv['predicates']
print(wv['predicates'])





