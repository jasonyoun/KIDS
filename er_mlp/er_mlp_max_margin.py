import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
print(__file__)
print(directory)
abs_path_er_mlp= os.path.join(directory, '..')
sys.path.insert(0, abs_path_er_mlp)
abs_path_metrics= os.path.join(directory, 'utils')
sys.path.insert(0, abs_path_metrics)
if directory != '':
    directory = directory+'/'
# print(directory)
#sys.path.insert(0, '../data')
import tensorflow as tf
from sklearn import utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import random
from tensorflow.python import debug as tf_debug
from scipy import interp
import random
from data_processor import DataProcessor
from er_mlp import ERMLP
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
# from sklearn.model_selection import StratifiedKFold
#from sklearn.cross_validation import StratifiedKFold



def run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, LAYER_SIZE,TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS, DROP_OUT_PERCENT,DATA_PATH, SAVE_MODEL=False, MODEL_SAVE_DIRECTORY=None  ):
    # numerically represent the entities, predicates, and words
    processor = DataProcessor()

    train_df = processor.load(DATA_PATH+'train.txt')
    test_df = processor.load(DATA_PATH+'test.txt')
    dev_df = processor.load(DATA_PATH+'dev.txt')

    print("machine translation...")
    indexed_entities, num_entity_words, entity_dic,indexed_predicates, indexed_pred_word_embeddings, pred_dic,num_pred_words,num_entity_words = None,None,None,None,None,None,None,None
    if WORD_EMBEDDING:
        indexed_entities, num_entity_words, entity_dic = processor.machine_translate_using_word(DATA_PATH+'/entities.txt',EMBEDDING_SIZE)
        indexed_predicates, num_pred_words, pred_dic = processor.machine_translate_using_word(DATA_PATH+'/relations.txt',EMBEDDING_SIZE)
    else:
        entity_dic = processor.machine_translate(DATA_PATH+'/entities.txt',EMBEDDING_SIZE)
        pred_dic = processor.machine_translate(DATA_PATH+'/relations.txt',EMBEDDING_SIZE)

    # numerically represent the data 
    print("Index:")
    indexed_train_data = processor.create_indexed_triplets_training(train_df.as_matrix(),entity_dic,pred_dic )
    indexed_test_data = processor.create_indexed_triplets_test(test_df.as_matrix(),entity_dic,pred_dic )
    indexed_dev_data = processor.create_indexed_triplets_test(dev_df.as_matrix(),entity_dic,pred_dic )
    indexed_test_data[:,3][indexed_test_data[:,3] == 0] = -1
    indexed_dev_data[:,3][indexed_dev_data[:,3] == 0] = -1


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
        'indexed_entities':indexed_entities,
        'indexed_predicates': indexed_predicates, 
        'learning_rate':LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'add_layers': ADD_LAYERS,
        'act_function':ACT_FUNCTION,
        'drop_out_percent': DROP_OUT_PERCENT
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

    tf.add_to_collection('training_predictions', training_predictions)
    tf.add_to_collection('predictions', predictions)

    print('calculate cost')
    cost = er_mlp.loss(training_predictions)

    tf.add_to_collection('cost', cost)

    print('optimizer')
    if OPTIMIZER == 0:
        print('adagrad')
        optimizer = er_mlp.train_adagrad(cost)
    else:
        print('adam')
        optimizer = er_mlp.train_adam(cost)

    tf.add_to_collection('optimizer', optimizer)


    print("initialize tensor variables")
    init_all = tf.global_variables_initializer()

    print("begin tensor seesion")
    sess = tf.Session()

    saver = tf.train.Saver()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(init_all)


    data_train = indexed_data_training[:,:3]


    def determine_threshold(indexed_data_dev, f1=False):
        # use the dev set to compute the best threshold for classification
        data_dev = indexed_data_dev[:,:3]
        predicates_dev = indexed_data_dev[:,1]
        labels_dev = indexed_data_dev[:,3]
        labels_dev = labels_dev.reshape((np.shape(labels_dev)[0],1))
        predictions_dev = sess.run(predictions, feed_dict={triplets: data_dev, y: labels_dev})
        threshold = er_mlp.compute_threshold(predictions_dev,labels_dev,predicates_dev,f1,cross_margin=True)
        return threshold
    
    def test_model( indexed_data_test,threshold, _type='current'):
        
        data_test = indexed_data_test[:,:3]
        labels_test = indexed_data_test[:,3]
        labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
        predicates_test = indexed_data_test[:,1]
        predictions_list_test = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})
        mean_average_precision_test = pr_stats(NUM_PREDS, labels_test, predictions_list_test,predicates_test)
        roc_auc_test = roc_auc_stats(NUM_PREDS, labels_test, predictions_list_test,predicates_test)
        classifications_test = er_mlp.classify(predictions_list_test,threshold, predicates_test,, cross_margin=True)
        classifications_test = np.array(classifications_test).astype(int)
        labels_test = labels_test.astype(int)
        fl_measure_test = f1_score(labels_test, classifications_test)
        accuracy_test = accuracy_score(labels_test, classifications_test)

        print(_type+" test mean average precision:"+ str(mean_average_precision_test))
        print(_type+" test f1 measure:"+ str(fl_measure_test))
        print(_type+" test accuracy:"+ str(accuracy_test))
        print(_type+" test roc auc:"+ str(roc_auc_test))


    iter_list = []
    cost_list = []
    iteration = 0
    current_cost = 0.
    current_accuracy = None
    current_threshold = None
    print("Begin training...")
    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0.
        total_batch = int(data_train.shape[0] / BATCH_SIZE)
        for i in range(total_batch):
            batch_xs = er_mlp.get_training_batch_with_corrupted(data_train)
            flip = bool(random.getrandbits(1))
            _, current_cost= sess.run([optimizer, cost], feed_dict={training_triplets: batch_xs, flip_placeholder: flip})
            avg_cost +=current_cost/total_batch
            # print(current_cost)
            cost_list.append(current_cost)
            iter_list.append(iteration)
            iteration+=1
        # Display progress
        if epoch % DISPLAY_STEP == 0:
            thresholds = determine_threshold(indexed_dev_data)
            test_model( indexed_test_data, thresholds, _type='current')
            print ("Epoch: %03d/%03d cost: %.9f - current_cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost,current_cost ))

    print("determine threshold for classification")
    
    thresholds = determine_threshold(indexed_dev_data)
    test_model(indexed_train_data, indexed_test_data, thresholds, _type='final')


    if SAVE_MODEL:


        saver.save(sess,MODEL_SAVE_DIRECTORY+'/model')
        print('model saved in: '+MODEL_SAVE_DIRECTORY)
        save_object = {
            'thresholds':thresholds,
            'entity_dic': entity_dic,
            'pred_dic': pred_dic
        }
        if WORD_EMBEDDING:
            save_object['indexed_entities'] = indexed_entities
            save_object['indexed_predicates'] = indexed_predicates
            save_object['num_pred_words'] = num_pred_words
            save_object['num_entity_words'] = num_entity_words
        with open(MODEL_SAVE_DIRECTORY+'/params.pkl', 'wb') as output:
            pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    WORD_EMBEDDING = False
    DATA_TYPE = 'freebase'
    EMBEDDING_SIZE = 60 # size of each embeddings
    LAYER_SIZE = 60 # number of columns in the first layer
    TRAINING_EPOCHS = 5 
    BATCH_SIZE = 500
    LEARNING_RATE = 0.01  
    DISPLAY_STEP = 1
    CORRUPT_SIZE = 10
    LAMBDA = 0.0001
    OPTIMIZER = 1
    ACT_FUNCTION = 0
    ADD_LAYERS=0
    DROP_OUT_PERCENT=0.1
    DATA_PATH = '/Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/archive/data/raw/freebase/'
    run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, \
        LAYER_SIZE, TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS)










