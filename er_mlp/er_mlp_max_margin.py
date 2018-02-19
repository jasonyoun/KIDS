import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
print(__file__)
print(directory)
abs_path_to_data = os.path.join(directory, '..', 'data')
sys.path.insert(0, abs_path_to_data)
if directory != '':
    directory = directory+'/'
# print(directory)
#sys.path.insert(0, '../data')
import tensorflow as tf
from sklearn import utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import random
from tensorflow.python import debug as tf_debug
from scipy import interp

from data_processor import DataProcessor
from er_mlp import ERMLP



def run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, LAYER_SIZE,TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS ):
    # numerically represent the entities, predicates, and words
    processor = DataProcessor()
    print("machine translation...")
    indexed_entities, num_entity_words, entity_dic,indexed_predicates, indexed_pred_word_embeddings, pred_dic,num_pred_words,num_entity_words = None,None,None,None,None,None,None,None
    if WORD_EMBEDDING:
        indexed_entities, num_entity_words, entity_dic = processor.machine_translate_using_word(directory+'../data/raw/{}/entities.txt'.format(DATA_TYPE),EMBEDDING_SIZE, directory+'../data/raw/{}/initEmbed.mat'.format(DATA_TYPE) )
        indexed_predicates, num_pred_words, pred_dic = processor.machine_translate_using_word(directory+'../data/raw/{}/relations.txt'.format(DATA_TYPE),EMBEDDING_SIZE)
    else:
        entity_dic = processor.machine_translate(directory+'../data/raw/{}/entities.txt'.format(DATA_TYPE),EMBEDDING_SIZE)
        pred_dic = processor.machine_translate(directory+'../data/raw/{}/relations.txt'.format(DATA_TYPE),EMBEDDING_SIZE)

    # load the data
    train_df = processor.load(directory+'../data/raw/{}/train.txt'.format(DATA_TYPE))
    test_df = processor.load(directory+'../data/raw/{}/test.txt'.format(DATA_TYPE))
    dev_df = processor.load(directory+'../data/raw/{}/dev.txt'.format(DATA_TYPE))

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
        'add_layers': ADD_LAYERS
    }

    er_mlp = ERMLP(er_mlp_params)

    triplets, weights, biases, constants, y = er_mlp.create_tensor_terms()

    # The training triplets: subject, predicate, object, corrupted_entity
    training_triplets = tf.placeholder(tf.int32, shape=(None, 4))

    # A boolean to determine if we want to corrupt the head or tail
    flip_placeholder = tf.placeholder(tf.bool)

    training_predictions = er_mlp.inference_for_max_margin_training(training_triplets, weights, biases, constants, flip_placeholder, ACT_FUNCTION, ADD_LAYERS)

    print('network for predictions')
    predictions = er_mlp.inference(triplets, weights, biases, constants, ACT_FUNCTION, ADD_LAYERS)

    print('calculate cost')
    cost = er_mlp.loss(training_predictions)

    print('optimizer')
    if OPTIMIZER == 0:
        print('adagrad')
        optimizer = er_mlp.train_adagrad(cost)
    else:
        print('adam')
        optimizer = er_mlp.train_adam(cost)






    print("initialize tensor variables")
    init_all = tf.global_variables_initializer()

    print("begin tensor seesion")
    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(init_all)


    data_train = indexed_data_training[:,:3]

    def roc_auc_stats( Y, predictions,predicates, params):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        predicates_included = []
        for i in range(params['num_preds']):
            predicate_indices = np.where(predicates == i)[0]
            if np.shape(predicate_indices)[0] == 0:
                print('inside')
                continue
            else:
                predicates_included.append(i)
            predicate_predictions = predictions[predicate_indices]
            predicate_labels = Y[predicate_indices]
            fpr[i], tpr[i] , _ = roc_curve(predicate_labels.ravel(), predicate_predictions.ravel())
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in predicates_included]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in predicates_included:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(predicates_included)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        return roc_auc["macro"]

    def determine_threshold(indexed_data_dev):
        # use the dev set to compute the best threshold for classification
        data_dev = indexed_data_dev[:,:3]
        predicates_dev = indexed_data_dev[:,1]
        labels_dev = indexed_data_dev[:,3]
        labels_dev = labels_dev.reshape((np.shape(labels_dev)[0],1))
        predictions_dev = sess.run(predictions, feed_dict={triplets: data_dev, y: labels_dev})
        threshold = er_mlp.compute_threshold(predictions_dev,labels_dev,predicates_dev)
        return threshold

    def test_model(indexed_data_test,threshold):
        data_test = indexed_data_test[:,:3]
        labels_test = indexed_data_test[:,3]
        labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
        predicates_test = indexed_data_test[:,1]
        predictions_list = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})
        classifications = er_mlp.classify(predictions_list,threshold, predicates_test)
        accuracy = sum(1 for x,y in zip(labels_test,classifications) if x == y) / len(labels_test)
        return accuracy, classifications, labels_test, predicates_test, predictions_list


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
            current_threshold = determine_threshold(indexed_data_dev)
            current_accuracy, c_, l_, p_, pr_ = test_model(indexed_data_test, current_threshold)
            current_auc = roc_auc_stats(l_,pr_,p_, er_mlp_params)
            print ("Epoch: %03d/%03d cost: %.9f - current_cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost,current_cost ))
            print("current accuracy:"+ str(current_accuracy))
            print("current auc:"+ str(current_auc))

    print("determine threshold for classification")
    # use the dev set to compute the best threshold for classification
    threshold = determine_threshold(indexed_data_dev)
    


    print("test model")
    # Test the model by classifying each sample using the threshold determined by running the model
    # over the dev set
    accuracy, classifications, labels_test, predicates_test, predictions_list = test_model(indexed_data_test,threshold)
    print("overall accuracy:"+ str(accuracy))


    # find the accuracy for the baseline, which is the most often occuring class
    a = np.empty(np.shape(classifications))
    a.fill(-1)
    accuracy_b = sum(1 for x,y in zip(labels_test,a) if x == y) / len(a)
    print("baseline accuracy:")
    print(accuracy_b)

    if __name__ == "__main__":
        from plotter import Plotter
        plotter_params = {
            'batch_size': BATCH_SIZE,
            'training_epochs': TRAINING_EPOCHS,
            'data_type': DATA_TYPE,
            'num_entities':NUM_ENTITIES,
            'num_preds':NUM_PREDS,
            'pred_dic':pred_dic
        }
        plotter = Plotter(plotter_params)
        plotter.plot_roc(labels_test,predictions_list,predicates_test)
        plotter.plot_pr(labels_test,predictions_list,predicates_test)
        plotter.plot_cost(iter_list,cost_list)

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

    return accuracy, roc_auc_stats(labels_test,predictions_list,predicates_test, er_mlp_params)

if __name__ == "__main__":
    WORD_EMBEDDING = False
    DATA_TYPE = 'ecoli'
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
    ADD_LAYERS=2
    run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, \
        LAYER_SIZE, TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS)










