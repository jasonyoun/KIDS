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
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
import argparse
if directory != '':
    directory = directory+'/'

config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description='build network')
parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')

args = parser.parse_args()
model_instance_dir='model_instance/'
model_save_dir=model_instance_dir+args.dir
configuration = model_save_dir+'/'+args.dir.replace('/','')+'.ini'

print('./'+configuration)
config.read('./'+configuration)
WORD_EMBEDDING = config.getboolean('DEFAULT','WORD_EMBEDDING')
MODEL_SAVE_DIRECTORY=model_save_dir
DATA_PATH=config['DEFAULT']['DATA_PATH']
WORD_EMBEDDING = config.getboolean('DEFAULT','WORD_EMBEDDING')
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
F1_FOR_THRESHOLD=config.getboolean('DEFAULT','F1_FOR_THRESHOLD')


print("begin tensor seesion")
with tf.Session() as sess:

    processor = DataProcessor()
    saver = tf.train.import_meta_graph(MODEL_SAVE_DIRECTORY+'/model.meta')
    saver.restore(sess, MODEL_SAVE_DIRECTORY+'/model')
    fn = open(MODEL_SAVE_DIRECTORY+'/params.pkl','rb')
    params = pickle.load(fn)
    entity_dic = params['entity_dic']
    pred_dic = params['pred_dic']

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

    dev_df = processor.load(DATA_PATH+'dev.txt')
    indexed_data_dev= processor.create_indexed_triplets_test(dev_df.as_matrix(),entity_dic,pred_dic )
    indexed_data_dev[:,3][indexed_data_dev[:,3] == -1] = 0
     
    data_dev = indexed_data_dev[:,:3]
    labels_dev = indexed_data_dev[:,3]
    labels_dev = labels_dev.reshape((np.shape(labels_dev)[0],1))
    predicates_dev = indexed_data_dev[:,1]
    predictions_list_dev = sess.run(predictions, feed_dict={triplets: data_dev, y: labels_dev})
    thresholds = er_mlp.compute_threshold(predictions_list_dev,labels_dev,predicates_dev,f1=F1_FOR_THRESHOLD)
    print('thresholds')
    print(thresholds)
    save_object = {
        'entity_dic': entity_dic,
        'pred_dic': pred_dic,
        'thresholds':thresholds
    }
    if hasattr(params, 'thresholds_calibrated'):
        save_object['thresholds_calibrated'] = params['thresholds_calibrated']

    if hasattr(params, 'calibrated_models'):
        save_object['calibrated_models'] = params['calibrated_models']


    if WORD_EMBEDDING:
        save_object['indexed_entities'] = indexed_entities
        save_object['indexed_predicates'] = indexed_predicates
        save_object['num_pred_words'] = num_pred_words
        save_object['num_entity_words'] = num_entity_words
    with open(MODEL_SAVE_DIRECTORY+'/params.pkl', 'wb') as output:
        pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)

