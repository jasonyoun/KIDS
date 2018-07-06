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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import random
from tensorflow.python import debug as tf_debug
from scipy import interp
import random
from er_mlp import ERMLP
abs_path_metrics= os.path.join(directory, '../../utils')
sys.path.insert(0, abs_path_metrics)
from data_processor import DataProcessor
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats, save_results
import argparse
if directory != '':
    directory = directory+'/'



config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description='evaluate')
parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')
parser.add_argument('--use_calibration',action='store_const',default=False,const=True)



args = parser.parse_args()
calibrated = args.use_calibration
model_instance_dir='model_instance/'
model_save_dir=model_instance_dir+args.dir
configuration = model_save_dir+'/'+args.dir.replace('/','')+'.ini'
print('./'+configuration)
config.read('./'+configuration)
WORD_EMBEDDING = config.getboolean('DEFAULT','WORD_EMBEDDING')
MODEL_SAVE_DIRECTORY=model_save_dir
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
LOG_REG_CALIBRATE= config.getboolean('DEFAULT','LOG_REG_CALIBRATE')

def calibrate_probabilties(predictions_list_test,num_preds,calibration_models,predicates_test,pred_dic):
    index=0
    for k,i in pred_dic.items():
        indices, = np.where(predicates_test == i)
        if np.shape(indices)[0]!=0 :
            predictions_predicate = predictions_list_test[indices]
            clf = calibration_models[i]
            if LOG_REG_CALIBRATE:
                p_calibrated = clf.predict_proba( predictions_predicate.reshape( -1, 1 ))[:,1]
            else:
                p_calibrated = clf.transform( predictions_predicate.ravel() )
            predictions_list_test[indices] = p_calibrated.reshape((np.shape(p_calibrated)[0],1))
            print(predictions_list_test[indices])
    return predictions_list_test

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
        thresholds = params['thresholds_calibrated']
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
        predictions_list_test= calibrate_probabilties(predictions_list_test,num_preds,calibration_models,predicates_test,pred_dic)



    mean_average_precision_test = pr_stats(num_preds, labels_test, predictions_list_test,predicates_test,pred_dic)
    roc_auc_test = roc_auc_stats(num_preds, labels_test, predictions_list_test,predicates_test,pred_dic)
    classifications_test = er_mlp.classify(predictions_list_test,thresholds, predicates_test)
    classifications_test = np.array(classifications_test).astype(int)
    labels_test = labels_test.astype(int)
    fl_measure_test = f1_score(labels_test, classifications_test)
    accuracy_test = accuracy_score(labels_test, classifications_test)
    confusion_test = confusion_matrix(labels_test, classifications_test)
    precision_test = precision_score(labels_test, classifications_test)
    recall_test = recall_score(labels_test, classifications_test)
    calib_file_name='_calibrated' if calibrated else '_not_calibrated'
    plot_pr(len(pred_dic), labels_test, predictions_list_test,predicates_test,pred_dic, MODEL_SAVE_DIRECTORY+'/test/',name_of_file='er_mlp'+calib_file_name)
    plot_roc(len(pred_dic), labels_test, predictions_list_test,predicates_test,pred_dic, MODEL_SAVE_DIRECTORY+'/test/',name_of_file='er_mlp'+calib_file_name)
    results = {}
    results['overall'] = {
        'map':mean_average_precision_test,
        'roc_auc':roc_auc_test,
        'f1':fl_measure_test,
        'accuracy': accuracy_test,
        'cm': confusion_test,
        'precision': precision_test,
        'recall': recall_test
    }  
    results['predicate'] = {}
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
            precision_predicate = precision_score(labels_predicate, classifications_predicate)
            recall_predicate = recall_score(labels_predicate, classifications_predicate)
            print(" - test f1 measure for "+pred_name+ ":"+ str(fl_measure_predicate))
            print(" - test accuracy for "+pred_name+ ":"+ str(accuracy_predicate))
            print(" - test precision for "+pred_name+ ":"+ str(precision_predicate))
            print(" - test recall for "+pred_name+ ":"+ str(recall_predicate))
            print(" - test confusion matrix for "+pred_name+ ":")
            print(confusion_predicate)
            print(" ")
            predicate_predictions = predictions_list_test[indices]
            fpr_pred, tpr_pred , _ = roc_curve(labels_predicate.ravel(), predicate_predictions.ravel())
            roc_auc_pred = auc(fpr_pred, tpr_pred)
            ap_pred = average_precision_score(labels_predicate.ravel(), predicate_predictions.ravel())
            results['predicate'][pred_name] = {
                'map':ap_pred,
                'roc_auc':roc_auc_pred,
                'f1':fl_measure_predicate,
                'accuracy': accuracy_predicate,
                'cm': confusion_predicate,
                'precision': precision_predicate,
                'recall': recall_predicate
            }  
    
    print("test mean average precision:"+ str(mean_average_precision_test))
    print("test f1 measure:"+ str(fl_measure_test))
    print("test accuracy:"+ str(accuracy_test))
    print("test roc auc:"+ str(roc_auc_test))
    print("test precision:"+ str(precision_test))
    print("test recall:"+ str(recall_test))
    print("test confusion matrix:")
    print(confusion_test)
    print("thresholds: ")
    print(thresholds)
    save_results(results,MODEL_SAVE_DIRECTORY+'/test/')


_file =   MODEL_SAVE_DIRECTORY+'/test/classifications_er_mlp.txt'
with open(_file, 'w') as t_f:
    for row in classifications_test:
        t_f.write(str(row)+'\n')
