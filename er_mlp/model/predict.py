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

config = configparser.ConfigParser()
configuration = sys.argv[1]+'.ini'
calibrated = False
if len(sys.argv)>3:
    if sys.argv[3] == 'use_calibration':
        calibrated=True

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
PREDICT_FOLDER = sys.argv[2]

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

    test_df = processor.load(PREDICT_FOLDER+'/data.txt')
    print(test_df.shape)
    indexed_data_test = processor.create_indexed_triplets_training(test_df.as_matrix(),entity_dic,pred_dic )
    if (test_df.shape[1]==4):
        indexed_data_test = processor.create_indexed_triplets_test(test_df.as_matrix(),entity_dic,pred_dic )
        print(np.shape(indexed_data_test))
    data_test = indexed_data_test[:,:3]
    predicates_test = indexed_data_test[:,1]
    predictions_list_test = sess.run(predictions, feed_dict={triplets: data_test})
    
    if calibrated:
        predictions_list_test,thresholds, thresholds_dic = calibrate_probabilties(predictions_list_test,num_preds,calibration_models,predicates_test,pred_dic)

    print(np.shape(predictions_list_test))
    classifications_test = er_mlp.classify(predictions_list_test,thresholds, predicates_test,pred_dic=pred_dic,thresholds_dic=thresholds_dic)
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

    with open(PREDICT_FOLDER+"/predictions.txt", 'w') as _file:
        for i in range(np.shape(c)[0]):
            if (test_df.shape[1]==4):
                _file.write("predicate: "+str(predicates_test[i])+"\tclassification: "+str(int(c[i][0]))+ '\tprediction: '+str(c[i][1])+'\tlabel: '+str(int(c[i][2]))+'\n' )
            else:
                _file.write("predicate: "+str(predicates_test[i])+"\tclassification: "+str(int(c[i][0]))+ '\tprediction: '+str(c[i][1])+'\n' )
