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
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats,plot_cost
import argparse
if directory != '':
    directory = directory+'/'

config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description='Continue training')
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
DATA_TYPE = config['DEFAULT']['DATA_TYPE']
TRAINING_EPOCHS = config.getint('DEFAULT','CONTINUE_TRAINING_EPOCHS')
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
TRAIN_FILE = config['DEFAULT']['TRAIN_FILE']
SAVE_MODEL=config.getboolean('DEFAULT','SAVE_MODEL')
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
    #Placeholder
    #Placeholder_1
    training_triplets = graph.get_tensor_by_name("training_triplets:0") 
    flip_placeholder = graph.get_tensor_by_name("flip_placeholder:0")
    # training_triplets = None
    optimizer = graph.get_collection("optimizer")[0]
    cost = graph.get_collection("cost")[0]
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

    train_df = processor.load(DATA_PATH+TRAIN_FILE)

    indexed_train_data = processor.create_indexed_triplets_test(train_df.as_matrix(),entity_dic,pred_dic )
    data_train = indexed_train_data[indexed_train_data[:,3] == 1  ]
    data_train = data_train[:,:3]
    dev_df = processor.load(DATA_PATH+'dev.txt')
    indexed_dev_data = processor.create_indexed_triplets_test(dev_df.as_matrix(),entity_dic,pred_dic )
    indexed_dev_data[:,3][indexed_dev_data[:,3] == 0] = -1

    test_df = processor.load(DATA_PATH+'test.txt')
    indexed_test_data = processor.create_indexed_triplets_test(test_df.as_matrix(),entity_dic,pred_dic )
    indexed_test_data[:,3][indexed_test_data[:,3] == 0] = -1
     
    data_test = indexed_test_data[:,:3]
    labels_test = indexed_test_data[:,3]
    labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
    predicates_test = indexed_test_data[:,1]
    predictions_list_test = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})

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
        mean_average_precision_test = pr_stats(num_preds, labels_test, predictions_list_test,predicates_test,pred_dic)
        roc_auc_test = roc_auc_stats(num_preds, labels_test, predictions_list_test,predicates_test,pred_dic)
        classifications_test = er_mlp.classify(predictions_list_test,threshold, predicates_test, cross_margin=True)
        classifications_test = np.array(classifications_test).astype(int)
        confusion_test = confusion_matrix(labels_test, classifications_test)
        labels_test = labels_test.astype(int)
        fl_measure_test = f1_score(labels_test, classifications_test)
        accuracy_test = accuracy_score(labels_test, classifications_test)

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

        print(_type+" test mean average precision:"+ str(mean_average_precision_test))
        print(_type+" test f1 measure:"+ str(fl_measure_test))
        print(_type+" test accuracy:"+ str(accuracy_test))
        print(_type+" test roc auc:"+ str(roc_auc_test))
        print(_type+"test confusion matrix:")
        print(confusion_test)
    batches_per_epoch = np.floor(len(data_train) / BATCH_SIZE).astype(np.int16)
    iter_list = []
    cost_list = []
    iteration = 0
    current_cost = 0.
    current_accuracy = None
    current_threshold = None
    print("Begin training...")


    for epoch in range(TRAINING_EPOCHS):
        if epoch == 0:
            thresholds = determine_threshold(indexed_dev_data,f1=F1_FOR_THRESHOLD)
            test_model( indexed_test_data, thresholds, _type='current')
        avg_cost = 0.
        total_batch = int(data_train.shape[0] / BATCH_SIZE)
        for i in range(total_batch):
            batch_xs = er_mlp.get_training_batch_with_corrupted(data_train)
            flip = bool(random.getrandbits(1))
            # batch_xs = data_orch.get_next_training_batch(BATCH_SIZE,flip)
            _, current_cost= sess.run([optimizer, cost], feed_dict={training_triplets: batch_xs, flip_placeholder: flip})
            avg_cost +=current_cost/total_batch
            # print(current_cost)
            cost_list.append(current_cost)
            iter_list.append(iteration)
            iteration+=1
        # Display progress
        if epoch % DISPLAY_STEP == 0:
            thresholds = determine_threshold(indexed_dev_data,f1=F1_FOR_THRESHOLD)
            test_model( indexed_test_data, thresholds, _type='current')
            print ("Epoch: %03d/%03d cost: %.9f - current_cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost,current_cost ))
            print ("")
        # data_orch.reset_data_index()

    print("determine threshold for classification")
    
    thresholds = determine_threshold(indexed_dev_data,f1=F1_FOR_THRESHOLD)
    test_model( indexed_test_data, thresholds, _type='final')
    plot_cost(iter_list,cost_list,MODEL_SAVE_DIRECTORY)


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


