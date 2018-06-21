import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
abs_path_metrics= os.path.join(directory, '../../utils')
sys.path.insert(0, abs_path_metrics)
abs_path_data= os.path.join(directory, '../data_handler')
sys.path.insert(0, abs_path_data)
if directory != '':
    directory = directory+'/'
import tensorflow as tf
from sklearn import utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score, confusion_matrix
import random
from tensorflow.python import debug as tf_debug
from scipy import interp
import random
from data_processor import DataProcessor
from data_orchestrator_cm import DataOrchestrator
from er_mlp import ERMLP
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats, plot_cost



def run_model(params):
    # numerically represent the entities, predicates, and words
    processor = DataProcessor()

    train_df = processor.load(params['DATA_PATH']+params['TRAIN_FILE'])
    if len(train_df.columns)<4:
        train_df['one'] =1

    test_df = processor.load(params['DATA_PATH']+'test.txt')
    dev_df = processor.load(params['DATA_PATH']+'dev.txt')

    print("machine translation...")
    # indexed_entities, num_entity_words, entity_dic,indexed_predicates, indexed_pred_word_embeddings, pred_dic,num_pred_words,num_entity_words = None,None,None,None,None,None,None,None
    if params['WORD_EMBEDDING']:
        indexed_entities, num_entity_words, entity_dic = processor.machine_translate_using_word(params['DATA_PATH']+'/entities.txt',params['EMBEDDING_SIZE'])
        indexed_predicates, num_pred_words, pred_dic = processor.machine_translate_using_word(params['DATA_PATH']+'/relations.txt',params['EMBEDDING_SIZE'])
    else:
        entity_dic = processor.machine_translate(params['DATA_PATH']+'/entities.txt',params['EMBEDDING_SIZE'])
        pred_dic = processor.machine_translate(params['DATA_PATH']+'/relations.txt',params['EMBEDDING_SIZE'])

    # numerically represent the data 
    print("Index:")
    indexed_train_data = processor.create_indexed_triplets_test(train_df.as_matrix(),entity_dic,pred_dic )
    indexed_test_data = processor.create_indexed_triplets_test(test_df.as_matrix(),entity_dic,pred_dic )
    indexed_dev_data = processor.create_indexed_triplets_test(dev_df.as_matrix(),entity_dic,pred_dic )

    print(np.shape(indexed_test_data))
    indexed_test_data[:,3][indexed_test_data[:,3] == 0] = -1
    np.random.shuffle(indexed_test_data)
    print(indexed_test_data)
    indexed_dev_data[:,3][indexed_dev_data[:,3] == 0] = -1


    NUM_ENTITIES = len(entity_dic)
    NUM_PREDS = len(pred_dic)

    er_mlp_params = {
        'word_embedding': params['WORD_EMBEDDING'],
        'embedding_size': params['EMBEDDING_SIZE'],
        'layer_size': params['LAYER_SIZE'],
        'corrupt_size': params['CORRUPT_SIZE'],
        'lambda': params['LAMBDA'],
        'num_entities':NUM_ENTITIES,
        'num_preds':NUM_PREDS,
        'num_entity_words':num_entity_words,
        'num_pred_words':num_pred_words,
        'indexed_entities':indexed_entities,
        'indexed_predicates': indexed_predicates, 
        'learning_rate':params['LEARNING_RATE'],
        'batch_size': params['BATCH_SIZE'],
        'add_layers': params['ADD_LAYERS'],
        'act_function':params['ACT_FUNCTION'],
        'drop_out_percent': params['DROP_OUT_PERCENT']
    }

    er_mlp = ERMLP(er_mlp_params)

    triplets, weights, biases, constants, y = er_mlp.create_tensor_terms()

    # The training triplets: subject, predicate, object, corrupted_entity
    training_triplets = tf.placeholder(tf.int32, shape=(None, 4),name='training_triplets')

    # A boolean to determine if we want to corrupt the head or tail
    flip_placeholder = tf.placeholder(tf.bool,name='flip_placeholder')

    training_predictions = er_mlp.inference_for_max_margin_training(training_triplets, weights, biases, constants, flip_placeholder)

    print('network for predictions')
    predictions = er_mlp.inference(triplets, weights, biases, constants)

    tf.add_to_collection('training_predictions', training_predictions)
    tf.add_to_collection('predictions', predictions)

    print('calculate cost')
    cost = er_mlp.loss(training_predictions)

    tf.add_to_collection('cost', cost)

    print('optimizer')
    if params['OPTIMIZER'] == 0:
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

    print(np.shape(indexed_train_data))
    data_train = indexed_train_data[indexed_train_data[:,3] == 1  ]
    data_train = data_train[:,:3]
    print(np.shape(data_train))
    batches_per_epoch = np.floor(len(data_train) / params['BATCH_SIZE']).astype(np.int16)


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
        # labels_test[:][labels_test[:] == -1] = 0
        predicates_test = indexed_data_test[:,1]
        predictions_list_test = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})
        mean_average_precision_test = pr_stats(NUM_PREDS, labels_test, predictions_list_test,predicates_test,pred_dic)
        roc_auc_test = roc_auc_stats(NUM_PREDS, labels_test, predictions_list_test,predicates_test,pred_dic)
        classifications_test = er_mlp.classify(predictions_list_test,threshold, predicates_test, cross_margin=True)
        classifications_test = np.array(classifications_test).astype(int)
        confusion_test = confusion_matrix(labels_test, classifications_test)
        # classifications_test[:][classifications_test[:] == -1] = 0
        labels_test = labels_test.astype(int)
        fl_measure_test = f1_score(labels_test, classifications_test)
        # print(classifications_test)
        # print(labels_test)
        accuracy_test = accuracy_score(labels_test, classifications_test)

        for i in range(NUM_PREDS):
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



    iter_list = []
    cost_list = []
    iteration = 0
    current_cost = 0.
    current_accuracy = None
    current_threshold = None
    print("Begin training...")


    for epoch in range(params['TRAINING_EPOCHS']):
        if epoch == 0:
            thresholds = determine_threshold(indexed_dev_data,f1=params['F1_FOR_THRESHOLD'])
            test_model( indexed_test_data, thresholds, _type='current')
        avg_cost = 0.
        total_batch = int(data_train.shape[0] / params['BATCH_SIZE'])
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
        if epoch % params['DISPLAY_STEP'] == 0:
            thresholds = determine_threshold(indexed_dev_data,f1=params['F1_FOR_THRESHOLD'])
            test_model( indexed_test_data, thresholds, _type='current')
            print ("Epoch: %03d/%03d cost: %.9f - current_cost: %.9f" % (epoch, params['TRAINING_EPOCHS'], avg_cost,current_cost ))
            print ("")
        # data_orch.reset_data_index()

    print("determine threshold for classification")
    
    thresholds = determine_threshold(indexed_dev_data,f1=params['F1_FOR_THRESHOLD'])
    test_model( indexed_test_data, thresholds, _type='final')
    plot_cost(iter_list,cost_list,params['MODEL_SAVE_DIRECTORY'])


    if params['SAVE_MODEL']:


        saver.save(sess,params['MODEL_SAVE_DIRECTORY']+'/model')
        print('model saved in: '+params['MODEL_SAVE_DIRECTORY'])
        save_object = {
            'thresholds':thresholds,
            'entity_dic': entity_dic,
            'pred_dic': pred_dic
        }
        if params['WORD_EMBEDDING']:
            save_object['indexed_entities'] = indexed_entities
            save_object['indexed_predicates'] = indexed_predicates
            save_object['num_pred_words'] = num_pred_words
            save_object['num_entity_words'] = num_entity_words
        with open(params['MODEL_SAVE_DIRECTORY']+'/params.pkl', 'wb') as output:
            pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)












