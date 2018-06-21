import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
print(__file__)
print(directory)
import configparser
abs_path_er_mlp= os.path.join(directory, '../er_mlp_imp')
sys.path.insert(0, abs_path_er_mlp)
abs_path_metrics= os.path.join(directory, '../../utils')
sys.path.insert(0, abs_path_metrics)
abs_path_data= os.path.join(directory, '../data_handler')
sys.path.insert(0, abs_path_data)
from data_processor import DataProcessor
if directory != '':
    directory = directory+'/'
import tensorflow as tf
from sklearn import utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import random
from tensorflow.python import debug as tf_debug
from scipy import interp
import random
from er_mlp import ERMLP
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
from os.path import basename
import argparse

config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description='evaluate')
parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')
parser.add_argument('--use_calibration',action='store_const',default=False,const=True)
parser.add_argument('--predict_file', required=True)



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
PREDICT_FILE = args.predict_file
LOG_REG_CALIBRATE= config.getboolean('DEFAULT','LOG_REG_CALIBRATE')

def calibrate_probabilties(predictions_list_test,num_preds,calibration_models,predicates_test,pred_dic):

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
    print(predictions_list_test)
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

    test_df = processor.load(DATA_PATH+'/'+PREDICT_FILE)
    print(test_df.shape)
    indexed_data_test = processor.create_indexed_triplets_training(test_df.as_matrix(),entity_dic,pred_dic )
    if (test_df.shape[1]==4):
        indexed_data_test = processor.create_indexed_triplets_test(test_df.as_matrix(),entity_dic,pred_dic )
        print(np.shape(indexed_data_test))
    data_test = indexed_data_test[:,:3]
    predicates_test = indexed_data_test[:,1]
    predictions_list_test = sess.run(predictions, feed_dict={triplets: data_test})
    
    if calibrated:
        predictions_list_test = calibrate_probabilties(predictions_list_test,num_preds,calibration_models,predicates_test,pred_dic)

    print(np.shape(predictions_list_test))
    classifications_test = er_mlp.classify(predictions_list_test,thresholds, predicates_test)
    classifications_test = np.array(classifications_test).astype(int)
    classifications_test = classifications_test.reshape((np.shape(classifications_test)[0],1))
    print(np.shape(classifications_test))
    c = np.dstack((classifications_test,predictions_list_test))
    c = np.squeeze(c)
    print(np.shape(c))
    if (test_df.shape[1]==4):
        labels_test = indexed_data_test[:,3]
        labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
        print(np.shape(labels_test))
        c = np.concatenate((c,labels_test),axis=1)
    predict_folder = os.path.splitext(PREDICT_FILE)[0]
    predict_folder = MODEL_SAVE_DIRECTORY+'/'+predict_folder
    if not os.path.exists(predict_folder):
        os.makedirs(predict_folder)
    with open(predict_folder+"/predictions.txt", 'w') as _file:
        for i in range(np.shape(c)[0]):
            if (test_df.shape[1]==4):
                _file.write("predicate: "+str(predicates_test[i])+"\tclassification: "+str(int(c[i][0]))+ '\tprediction: '+str(c[i][1])+'\tlabel: '+str(int(c[i][2]))+'\n' )
            else:
                _file.write("predicate: "+str(predicates_test[i])+"\tclassification: "+str(int(c[i][0]))+ '\tprediction: '+str(c[i][1])+'\n' )
