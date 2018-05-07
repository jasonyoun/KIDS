import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
print(__file__)
print(directory)
import configparser
abs_path_data= os.path.join(directory, '../data_handler')
sys.path.insert(0, abs_path_data)
abs_path_er_mlp= os.path.join(directory, '../er_mlp_imp')
sys.path.insert(0, abs_path_er_mlp)
import tensorflow as tf
from sklearn import utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score, confusion_matrix
import random
from tensorflow.python import debug as tf_debug
from scipy import interp
import random
from er_mlp import ERMLP
abs_path_metrics= os.path.join(directory, '../../utils')
sys.path.insert(0, abs_path_metrics)
from data_processor import DataProcessor
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
if directory != '':
    directory = directory+'/'

config = configparser.ConfigParser()
configuration = sys.argv[1]+'.ini'
calibrated = False
if len(sys.argv)>2:
    if sys.argv[2] == 'use_calibration':
        calibrated=True
print('./'+configuration)
config.read('./'+configuration)
WORD_EMBEDDING = config.getboolean('DEFAULT','WORD_EMBEDDING')
MODEL_SAVE_DIRECTORY=config['DEFAULT']['MODEL_SAVE_DIRECTORY']
DATA_PATH=config['DEFAULT']['DATA_PATH']
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

def calibrate_probabilties(predictions_list_test,num_preds,calibration_models,predicates_test,pred_dic):
    thresholds_dic = {}
    thresholds = []
    index=0
    for k,i in pred_dic.items():
        indices, = np.where(predicates_test == i)
        if np.shape(indices)[0]!=0 :
            predictions_predicate = predictions_list_test[indices]
            log_reg = calibration_models[i]
            p_calibrated = log_reg.predict_proba( predictions_predicate.reshape( -1, 1 ))[:,1]
            predictions_list_test[indices] = p_calibrated.reshape((np.shape(p_calibrated)[0],1))
            print(predictions_list_test[indices])
            thresholds_dic[i]=.5
            thresholds.append(.5)
        index+=1
    print(predictions_list_test)
    return predictions_list_test,thresholds,thresholds_dic

print("begin tensor seesion")
with tf.Session() as sess:

    processor = DataProcessor()
    saver = tf.train.import_meta_graph(MODEL_SAVE_DIRECTORY+'/model.meta')
    saver.restore(sess, MODEL_SAVE_DIRECTORY+'/model')
    fn = open(MODEL_SAVE_DIRECTORY+'/params.pkl','rb')
    params = pickle.load(fn)
    entity_dic = params['entity_dic']
    pred_dic = params['pred_dic']
    thresholds = params['thresholds']
    if calibrated:
        calibration_models = params['calibrated_models']
        print(calibration_models)
    num_preds = len(pred_dic)
    num_entities= len(entity_dic)
    graph = tf.get_default_graph()
    y = graph.get_tensor_by_name("y:0")
    triplets = graph.get_tensor_by_name("triplets:0")
    predictions = tf.get_collection('predictions')[0]


    er_mlp_params = {
        'word_embedding': WORD_EMBEDDING,
        'embedding_size': EMBEDDING_SIZE,
        'layer_size': LAYER_SIZE,
        'corrupt_size': CORRUPT_SIZE,
        'lambda': LAMBDA,
        'num_entities':num_entities,
        'num_preds':num_preds,
        'learning_rate':LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'add_layers': ADD_LAYERS,
        'act_function':ACT_FUNCTION,
        'drop_out_percent': DROP_OUT_PERCENT
    }

    if WORD_EMBEDDING:
        num_entity_words = params['num_entity_words']
        num_pred_words = params['num_pred_words']
        indexed_entities = params['indexed_entities']
        indexed_predicates = params['indexed_predicates']
        er_mlp_params['num_entity_words'] = num_entity_words
        er_mlp_params['num_pred_words'] = num_pred_words
        er_mlp_params['indexed_entities'] = indexed_entities
        er_mlp_params['indexed_predicates'] = indexed_predicates


    er_mlp = ERMLP(er_mlp_params)

    test_df = processor.load(DATA_PATH+'test.txt')
    indexed_data_test = processor.create_indexed_triplets_test(test_df.as_matrix(),entity_dic,pred_dic )
    indexed_data_test[:,3][indexed_data_test[:,3] == -1] = 0
     
    data_test = indexed_data_test[:,:3]
    labels_test = indexed_data_test[:,3]
    labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
    predicates_test = indexed_data_test[:,1]
    predictions_list_test = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})
    if calibrated:
        predictions_list_test,thresholds, thresholds_dic = calibrate_probabilties(predictions_list_test,num_preds,calibration_models,predicates_test,pred_dic)
    else:
        thresholds_dic=None



    mean_average_precision_test = pr_stats(num_preds, labels_test, predictions_list_test,predicates_test,pred_dic)
    roc_auc_test = roc_auc_stats(num_preds, labels_test, predictions_list_test,predicates_test,pred_dic)
    classifications_test = er_mlp.classify(predictions_list_test,thresholds, predicates_test,pred_dic=pred_dic,thresholds_dic=thresholds_dic)
    classifications_test = np.array(classifications_test).astype(int)
    labels_test = labels_test.astype(int)
    fl_measure_test = f1_score(labels_test, classifications_test)
    accuracy_test = accuracy_score(labels_test, classifications_test)
    confusion_test = confusion_matrix(labels_test, classifications_test)
    plot_pr(len(pred_dic), labels_test, predictions_list_test,predicates_test,pred_dic, MODEL_SAVE_DIRECTORY)
    plot_roc(len(pred_dic), labels_test, predictions_list_test,predicates_test,pred_dic, MODEL_SAVE_DIRECTORY)

    for i in range(num_preds):
        for key, value in pred_dic.items():
            if value == i:
                pred_name =key
        indices, = np.where(predicates_test == i)
        if np.shape(indices)[0]!=0:
            classifications_predicate = classifications_test[indices]
            labels_predicate = labels_test[indices]
            fl_measure_predicate = f1_score(labels_predicate, classifications_predicate)
            accuracy_predicate = accuracy_score(labels_predicate, classifications_predicate)
            confusion_predicate = confusion_matrix(labels_predicate, classifications_predicate)
            print(" - test f1 measure for "+pred_name+ ":"+ str(fl_measure_predicate))
            print(" - test accuracy for "+pred_name+ ":"+ str(accuracy_predicate))
            print(" - test confusion matrix for "+pred_name+ ":")
            print(confusion_predicate)
            print(" ")
    
    print("test mean average precision:"+ str(mean_average_precision_test))
    print("test f1 measure:"+ str(fl_measure_test))
    print("test accuracy:"+ str(accuracy_test))
    print("test roc auc:"+ str(roc_auc_test))
    print("test confusion matrix:")
    print(confusion_test)
    print("thresholds: ")
    print(thresholds)