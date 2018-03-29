from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
import shutil
directory = os.path.dirname(__file__)
abs_path_to_data = os.path.join(directory, '..', 'data')
sys.path.insert(0, abs_path_to_data)
if directory != '':
    directory = directory+'/'
import tensorflow as tf
from sklearn import utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import random
from tensorflow.python import debug as tf_debug
from scipy import interp

from data_processor import DataProcessor
from er_mlp import ERMLP
import time

MODELS_DIR = './models'


class Orchestrator:
    def __init__(self, model_id=None,params=None):
        self.init = False
        if params==None and model_id==None:
            print('invalid arguments')
        elif params==None and model_id!=None:
            self.load_configuration(model_id)
            self.model_id=model_id
        else:
            self.params = params
            self.init = True
            self.save_configuration(params)
        self.indexed_data ={}
        self.er_mlp = None

    def get_model_id(self):
        return self.model_id

    def save_configuration(self,params):
        current_time_millis = str(int(round(time.time() *1000)))
        self.model_id = current_time_millis
        if not os.path.exists(MODELS_DIR+'/'+self.model_id):
            os.makedirs(MODELS_DIR+'/'+self.model_id)
        model_path = os.path.join(MODELS_DIR,self.model_id )
        with open(model_path+'/'+self.model_id+'.pkl', 'wb') as output:
            pickle.dump(params, output, pickle.HIGHEST_PROTOCOL)
        self.params = params

    def load_configuration(self,model_id):
        self.model_id = model_id
        fn = open(MODELS_DIR+'/'+self.model_id+'/'+self.model_id+'.pkl','rb')
        self.params = pickle.load(fn)
        print(self.params)
     

    def load_data(self):
        processor = DataProcessor()
        print("machine translation...")
        print(self.params['word_embedding'])
        if self.params['word_embedding']:
            self.params['indexed_entities'], self.params['num_entity_words'],self.params['entity_dic'] = processor.machine_translate_using_word(directory+'../data/raw/{}/entities.txt'.format(self.params['data_type']),self.params['embedding_size'], directory+'../data/raw/{}/initEmbed.mat'.format(self.params['data_type']) )
            self.params['indexed_predicates'], self.params['num_pred_words'],self.params['pred_dic']= processor.machine_translate_using_word(directory+'../data/raw/{}/relations.txt'.format(self.params['data_type']),self.params['embedding_size'])
        else:
            self.params['entity_dic'] = processor.machine_translate(directory+'../data/raw/{}/entities.txt'.format(self.params['data_type']),self.params['embedding_size'])
            self.params['pred_dic'] = processor.machine_translate(directory+'../data/raw/{}/relations.txt'.format(self.params['data_type']),self.params['embedding_size'])

        processor = DataProcessor()
        print("machine translation...")

        # load the data
        train_df = processor.load(directory+'../data/raw/{}/train.txt'.format(self.params['data_type']))
        test_df = processor.load(directory+'../data/raw/{}/test.txt'.format(self.params['data_type']))
        dev_df = processor.load(directory+'../data/raw/{}/dev.txt'.format(self.params['data_type']))

        # numerically represent the data 
        print("Index:")
        print(" - training complete")
        self.indexed_data['train'] = processor.create_indexed_triplets_training(train_df.as_matrix(),self.params['entity_dic'],self.params['pred_dic'] )
        print(" - dev complete")
        self.indexed_data['dev']= processor.create_indexed_triplets_test(dev_df.as_matrix(),self.params['entity_dic'],self.params['pred_dic'] )
        print(" - test complete")
        self.indexed_data['test'] = processor.create_indexed_triplets_test(test_df.as_matrix(),self.params['entity_dic'],self.params['pred_dic'] )


        self.params['num_entities'] = len(self.params['entity_dic'])
        self.params['num_preds'] = len(self.params['pred_dic'])

        print('data loaded')

    def create_network(self):

        self.er_mlp = ERMLP(self.params)

        triplets, weights, biases, constants, y = self.er_mlp.create_tensor_terms()

        # The training triplets: subject, predicate, object, corrupted_entity
        training_triplets = tf.placeholder(tf.int32, shape=(None, 4), name='training_triplets')

        # A boolean to determine if we want to corrupt the head or tail
        flip_placeholder = tf.placeholder(tf.bool, name='flip_placeholder')

        training_predictions = self.er_mlp.inference_for_max_margin_training(training_triplets, weights, biases, constants, flip_placeholder)

        print('network for predictions')
        predictions = self.er_mlp.inference(triplets, weights, biases, constants)

        print('calculate cost')
        cost = self.er_mlp.loss(training_predictions)

        print('optimizer')
        if self.params['optimizer'] == 0:
            print('adagrad')
            optimizer = self.er_mlp.train_adagrad(cost)
        else:
            print('adam')
            optimizer = self.er_mlp.train_adam(cost)

        print("initialize tensor variables")
        init_all = tf.global_variables_initializer()

        saver = tf.train.Saver()
        tf.add_to_collection('predictions', predictions)
        tf.add_to_collection('cost', cost)
        tf.add_to_collection('optimizer', optimizer)

        print("begin tensor seesion")
        sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if self.init:
            sess.run(init_all)
            saver.save(sess,MODELS_DIR+'/'+self.model_id+'/'+self.model_id)
            self.init=False

    def train_network(self, dataset='train'):

        print("begin tensor seesion")
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(MODELS_DIR+'/'+self.model_id+'/'+self.model_id+'.meta')
            saver.restore(sess, MODELS_DIR+'/'+self.model_id+'/'+self.model_id)
            graph = tf.get_default_graph()
            training_triplets = graph.get_tensor_by_name("training_triplets:0")
            flip_placeholder = graph.get_tensor_by_name("flip_placeholder:0")
            y = graph.get_tensor_by_name("y:0")
            triplets = graph.get_tensor_by_name("triplets:0")
            optimizer =  tf.get_collection('optimizer')[0]
            cost =  tf.get_collection('cost')[0]
            predictions = tf.get_collection('predictions')[0]
            data_train = self.indexed_data[dataset][:,:3]
            indexed_data_dev = self.indexed_data['dev']
            indexed_data_test = self.indexed_data['test']

            iter_list = []
            cost_list = []
            iteration = 0
            current_cost = 0.
            current_accuracy = None
            current_threshold = None
            print("Begin training...")
            for epoch in range(self.params['training_epochs']):
                avg_cost = 0.
                total_batch = int(data_train.shape[0] / self.params['batch_size'])
                for i in range(total_batch):
                    batch_xs = self.er_mlp.get_training_batch_with_corrupted(data_train)
                    flip = bool(random.getrandbits(1))
                    _, current_cost= sess.run([optimizer, cost], feed_dict={training_triplets: batch_xs, flip_placeholder: flip})
                    avg_cost +=current_cost/total_batch
                    cost_list.append(current_cost)
                    iter_list.append(iteration)
                    iteration+=1
                if epoch % self.params['display_step'] == 0:
                    current_threshold = self.determine_threshold(indexed_data_dev, sess, y, triplets, predictions)
                    current_accuracy, c_, l_, p_, pr_ = self.test_model(indexed_data_test, current_threshold, sess, y, triplets, predictions)
                    current_auc = self.roc_auc_stats(l_,pr_,p_, self.params)
                    print ("Epoch: %03d/%03d cost: %.9f - current_cost: %.9f" % (epoch, self.params['training_epochs'], avg_cost,current_cost ))
                    print("current accuracy:"+ str(current_accuracy))
                    print("current auc:"+ str(current_auc))

            saver.save(sess,MODELS_DIR+'/'+self.model_id+'/'+self.model_id)

    def eval_network(self, dataset='test'):
        print("begin tensor seesion")
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(MODELS_DIR+'/'+self.model_id+'/'+self.model_id+'.meta')
            saver.restore(sess, MODELS_DIR+'/'+self.model_id+'/'+self.model_id)
            graph = tf.get_default_graph()
            y = graph.get_tensor_by_name("y:0")
            triplets = graph.get_tensor_by_name("triplets:0")
            predictions = tf.get_collection('predictions')[0]

            threshold = self.determine_threshold(self.indexed_data['dev'], sess, y, triplets, predictions)

            accuracy, c_, l_, p_, pr_ = self.test_model(self.indexed_data[dataset], threshold, sess, y, triplets, predictions)
            auc = self.roc_auc_stats(l_,pr_,p_, self.params)
            return accuracy, auc

    def predict(self, data):
        processor = DataProcessor()
        indexed_data = processor.create_indexed_triplets_training(data,self.params['entity_dic'],self.params['pred_dic'] )
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(MODELS_DIR+'/'+self.model_id+'/'+self.model_id+'.meta')
            saver.restore(sess, MODELS_DIR+'/'+self.model_id+'/'+self.model_id)
            graph = tf.get_default_graph()
            y = graph.get_tensor_by_name("y:0")
            triplets = graph.get_tensor_by_name("triplets:0")
            predictions = tf.get_collection('predictions')[0]

            data_test = indexed_data[:,:3] 
            predictions_list = sess.run(predictions, feed_dict={triplets: data_test})
            return predictions_list

    def roc_auc_stats(self, Y, predictions,predicates, params):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        predicates_included = []
        for i in range(params['num_preds']):
            predicate_indices = np.where(predicates == i)[0]
            if np.shape(predicate_indices)[0] == 0:
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

    def determine_threshold(self,indexed_data_dev, sess, y, triplets, predictions):
        # use the dev set to compute the best threshold for classification
        data_dev = indexed_data_dev[:,:3]
        predicates_dev = indexed_data_dev[:,1]
        labels_dev = indexed_data_dev[:,3]
        labels_dev = labels_dev.reshape((np.shape(labels_dev)[0],1))
        predictions_dev = sess.run(predictions, feed_dict={triplets: data_dev, y: labels_dev})
        threshold = self.er_mlp.compute_threshold(predictions_dev,labels_dev,predicates_dev)
        return threshold

    def test_model(self,indexed_data_test,threshold, sess, y, triplets, predictions):
        data_test = indexed_data_test[:,:3]
        labels_test = indexed_data_test[:,3]
        labels_test = labels_test.reshape((np.shape(labels_test)[0],1))
        predicates_test = indexed_data_test[:,1]
        predictions_list = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})
        classifications = self.er_mlp.classify(predictions_list,threshold, predicates_test)
        accuracy = sum(1 for x,y in zip(labels_test,classifications) if x == y) / len(labels_test)
        return accuracy, classifications, labels_test, predicates_test, predictions_list

    def delete_model(self):
        shutil.rmtree(MODELS_DIR+'/'+self.model_id) 




