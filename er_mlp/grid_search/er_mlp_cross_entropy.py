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
if directory != '':
    directory = directory+'/'
# print(directory)
#sys.path.insert(0, '../data')
import tensorflow as tf
from sklearn import utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import random
from tensorflow.python import debug as tf_debug
from scipy import interp
import random
from data_processor import DataProcessor
from er_mlp import ERMLP
from sklearn.model_selection import StratifiedKFold
#from sklearn.cross_validation import StratifiedKFold

SPLITS=2
OVER_SAMPLE=True
WORD_EMBEDDING = False


def over_sample(X, Y):
    s = X[:,0]
    s = s.reshape((np.shape(s)[0],1))
    p = X[:,1]
    p = p.reshape((np.shape(p)[0],1))
    o = X[:,2]
    o = o.reshape((np.shape(o)[0],1))
    t = Y.reshape((np.shape(Y)[0],1))
    combined_data = np.concatenate((s,p,o,t), axis=1)
    positive_samples = combined_data[combined_data[:,3] == 1]
    negative_samples = combined_data[combined_data[:,3] == 0]
    allIdx = np.array(range(0,np.shape(positive_samples)[0]))
    idx = np.random.choice(allIdx,size=np.shape(negative_samples)[0],replace=True)
    os_positive_samples = positive_samples[idx]
    combined_data = np.concatenate((os_positive_samples,negative_samples), axis=0)
    ret_X = combined_data[:,:3]
    ret_Y = combined_data[:,3]
    print(np.shape(positive_samples))
    print(np.shape(os_positive_samples))
    print(np.shape(negative_samples))
    print(np.shape(combined_data))
    return ret_X,ret_Y

def pr_stats(num_preds, Y, predictions,predicates):
    baseline = np.zeros(np.shape(predictions))
    baseline_precision, baseline_recall , _ = precision_recall_curve(Y.ravel(), baseline.ravel())
    baseline_aucPR = auc(baseline_recall, baseline_precision)

    precision = dict()
    recall = dict()
    aucPR = dict()
    ap = dict()
    predicates_included = []
    sum_ap = 0.0
    for i in range(num_preds):
        predicate_indices = np.where(predicates == i)[0]
        if np.shape(predicate_indices)[0] == 0:
            print('inside')
            continue
        else:
            predicates_included.append(i)
        predicate_predictions = predictions[predicate_indices]
        predicate_labels = Y[predicate_indices]
        precision[i], recall[i] , _ = precision_recall_curve(predicate_labels.ravel(), predicate_predictions.ravel())
        ap[i] = average_precision_score(predicate_labels.ravel(), predicate_predictions.ravel())
        sum_ap+=ap[i]
        aucPR[i] = auc(recall[i], precision[i])
    mean_average_precision = sum_ap/len(predicates_included)
    return mean_average_precision
def run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, LAYER_SIZE,TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS, DROP_OUT_PERCENT,DATA_PATH ):

    processor = DataProcessor()
    # load the data
    df = processor.load(DATA_PATH+'data.txt')
    # numerically represent the entities, predicates, and words
    entity_dic = processor.create_entity_dic(df.as_matrix())
    pred_dic = processor.create_relation_dic(df.as_matrix())
    print("machine translation...")
    # indexed_entities, num_entity_words, entity_dic,indexed_predicates, indexed_pred_word_embeddings, pred_dic,num_pred_words,num_entity_words = None,None,None,None,None,None,None,None
    # if WORD_EMBEDDING:
    #     indexed_entities, num_entity_words, entity_dic = processor.machine_translate_using_word(directory+'../data/raw/{}/entities.txt'.format(DATA_TYPE),EMBEDDING_SIZE, directory+'../data/raw/{}/initEmbed.mat'.format(DATA_TYPE) )
    #     indexed_predicates, num_pred_words, pred_dic = processor.machine_translate_using_word(directory+'../data/raw/{}/relations.txt'.format(DATA_TYPE),EMBEDDING_SIZE)
    # else:
    #     entity_dic = processor.machine_translate(directory+'../data/raw/{}/entities.txt'.format(DATA_TYPE),EMBEDDING_SIZE)
    #     pred_dic = processor.machine_translate(directory+'../data/raw/{}/relations.txt'.format(DATA_TYPE),EMBEDDING_SIZE)


    # numerically represent the data 
    print("Index:")
    indexed_data = processor.create_indexed_triplets_test(df.as_matrix(),entity_dic,pred_dic )
    print(" - index training complete")
    skf = StratifiedKFold(n_splits=SPLITS, shuffle=True)

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
        'learning_rate':LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'add_layers': ADD_LAYERS,
        'act_function':ACT_FUNCTION,
        'drop_out_percent': DROP_OUT_PERCENT
    }
    X = indexed_data[:,:3]
    Y = indexed_data[:,3]
    Y[Y ==-1] = 0
    print('indexed_data')
    print(np.shape(indexed_data))
    labels = Y.reshape((np.shape(Y)[0],1))
    Y_scores = np.zeros(labels.shape)
    # skf = StratifiedKFold(Y, n_folds=SPLITS, shuffle=True)
    # for trainIdx, testIdx in skf:
    for trainIdx, testIdx in skf.split(X, Y):
        data_train = X[trainIdx]
        labels_train = Y[trainIdx]
        er_mlp = ERMLP(er_mlp_params)

        triplets, weights, biases, constants, y = er_mlp.create_tensor_terms()

        print('network for predictions')
        training_predictions = er_mlp.inference(triplets, weights, biases, constants, training=True)
        print('network for predictions')
        predictions = er_mlp.inference(triplets, weights, biases, constants)

        cost = er_mlp.loss_cross_entropy(training_predictions,y)

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

        if OVER_SAMPLE:
            data_train,labels_train = over_sample(data_train,labels_train)





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
                randidx = np.random.randint(int(data_train.shape[0]), size = BATCH_SIZE)

                batch_xs = data_train[randidx, :]
                batch_ys = labels_train[randidx]
                batch_ys = batch_ys.reshape((np.shape(batch_ys)[0],1))

                _, current_cost= sess.run([optimizer, cost], feed_dict={triplets: batch_xs, y:batch_ys})
                avg_cost +=current_cost/total_batch
                cost_list.append(current_cost)
                iter_list.append(iteration)
                iteration+=1
            # Display progress
            if epoch % DISPLAY_STEP == 0:
                data_test = X[testIdx]
                labels_test = Y[testIdx]
                labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
                predicates_test = X[testIdx][:,1]
                predictions_list = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})

                data_train_eval = X[trainIdx]
                labels_test_eval = Y[trainIdx]
                labels_test_eval = labels_test_eval.reshape((np.shape(labels_test_eval)[0],1))
                predicates_test_eval = X[trainIdx][:,1]
                predictions_list_eval = sess.run(predictions, feed_dict={triplets: data_train_eval, y: labels_test_eval})

                mean_average_precision = pr_stats(NUM_PREDS, labels_test, predictions_list,predicates_test)
                mean_average_precision_train = pr_stats(NUM_PREDS, labels_test_eval, predictions_list_eval,predicates_test_eval)
                print ("Epoch: %03d/%03d cost: %.9f - current_cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost,current_cost ))
                print("current test mean average precision:"+ str(mean_average_precision))
                print("current train mean average precision:"+ str(mean_average_precision_train))

        


        print("test model")
        # Test the model by classifying each sample using the threshold determined by running the model
        # over the dev set
        data_test = X[testIdx]
        labels_test = Y[testIdx]
        labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
        Y_scores[testIdx] = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})
    
    predicates_test = X[:,1]
    mean_average_precision = pr_stats(NUM_PREDS, labels, Y_scores,predicates_test)
    print("Overall mean average precision for k-fold: "+ str(mean_average_precision))


    # return accuracy, roc_auc_stats(labels_test,predictions_list,predicates_test, er_mlp_params)

if __name__ == "__main__":
    WORD_EMBEDDING = False
    DATA_TYPE = 'ecoli'
    EMBEDDING_SIZE = 60 # size of each embeddings
    LAYER_SIZE = 60 # number of columns in the first layer
    TRAINING_EPOCHS = 100
    BATCH_SIZE = 500
    LEARNING_RATE = 0.01  
    DISPLAY_STEP = 1
    CORRUPT_SIZE = 10
    LAMBDA = 0.0001
    OPTIMIZER = 1
    ACT_FUNCTION = 0
    ADD_LAYERS=2
    DROP_OUT_PERCENT = 0.5
    run_model(WORD_EMBEDDING,DATA_TYPE, EMBEDDING_SIZE, \
        LAYER_SIZE, TRAINING_EPOCHS, BATCH_SIZE, LEARNING_RATE, DISPLAY_STEP, CORRUPT_SIZE, LAMBDA, OPTIMIZER, ACT_FUNCTION, ADD_LAYERS, DROP_OUT_PERCENT)










