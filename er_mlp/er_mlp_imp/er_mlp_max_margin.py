"""
Filename: er_mlp_max_margin.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Construct ER MLP using max margin loss and
    perform train, evaluation, and test.

To-do:
    1. take _determine_threshold() and _test_model() out
    2. use params['dev_file'] and params['test_file'] instead of
        'dev.txt' and 'test.txt'
    3. create_indexed_triplets_test() should be changed to
        create_indexed_triplets() ?
    4. do we need to shuffle test data? I don't think so.
    5. probably remove _type from _test_model()
    6. maybe put _test_model into er_mlp.py like compute_threshold
    7. figure out why we set label to -1 here but to 0 in determine_thresholds.py
"""

import os
import random
import pickle
import logging as log
import numpy as np
import tensorflow as tf
from er_mlp import ERMLP
from data_processor import DataProcessor
from metrics import plot_cost, plot_map

def run_model(params, final_model=False):
    """
    Run the ER_MLP model using max margin loss.

    Inputs:
        params: dictionary containing different
            parameters to be used when running the model
    """
    # numerically represent the entities, predicates, and words
    processor = DataProcessor()

    # load train / dev / test data as dataframe
    train_df = processor.load(os.path.join(params['DATA_PATH'], params['TRAIN_FILE']))
    train_local_df = processor.load(os.path.join(params['DATA_PATH'], 'train_local.txt'))
    if not final_model:
        dev_df = processor.load(os.path.join(params['DATA_PATH'], 'dev.txt'))
        test_df = processor.load(os.path.join(params['DATA_PATH'], 'test.txt'))

    log.debug('train dataframe shape: %s', train_df.shape)
    log.debug('train_local dataframe shape: %s', train_local_df.shape)
    if not final_model:
        log.debug('dev dataframe shape: %s', dev_df.shape)
        log.debug('test dataframe shape: %s', test_df.shape)

    if len(train_df.columns) < 4:
        log.warning('Label (last column) is missing')
        train_df['one'] = 1

    if params['WORD_EMBEDDING']:
        indexed_entities, num_entity_words, entity_dic = processor.machine_translate_using_word(
            os.path.join(params['DATA_PATH'], 'entities.txt'))
        indexed_predicates, num_pred_words, pred_dic = processor.machine_translate_using_word(
            os.path.join(params['DATA_PATH'], 'relations.txt'))
    else:
        entity_dic = processor.machine_translate(os.path.join(params['DATA_PATH'], 'entities.txt'))
        pred_dic = processor.machine_translate(os.path.join(params['DATA_PATH'], 'relations.txt'))

    # numerically represent the data
    indexed_train_data = processor.create_indexed_triplets_test(train_df.values, entity_dic, pred_dic)
    indexed_train_local_data = processor.create_indexed_triplets_test(train_local_df.values, entity_dic, pred_dic)
    if not final_model:
        indexed_dev_data = processor.create_indexed_triplets_test(dev_df.values, entity_dic, pred_dic)
        indexed_test_data = processor.create_indexed_triplets_test(test_df.values, entity_dic, pred_dic)

    # change label from 0 to -1 for test / dev data
    if not final_model:
        indexed_train_local_data[:, 3][indexed_train_local_data[:, 3] == -1] = 0
        indexed_dev_data[:, 3][indexed_dev_data[:, 3] == -1] = 0
        indexed_test_data[:, 3][indexed_test_data[:, 3] == -1] = 0

        # shuffle test data
        np.random.shuffle(indexed_train_local_data)
        np.random.shuffle(indexed_test_data)

    # construct new parameter dictionary to be actually fed into the network
    er_mlp_params = {
        'word_embedding': params['WORD_EMBEDDING'],
        'embedding_size': params['EMBEDDING_SIZE'],
        'layer_size': params['LAYER_SIZE'],
        'corrupt_size': params['CORRUPT_SIZE'],
        'lambda': params['LAMBDA'],
        'num_entities': len(entity_dic),
        'num_preds': len(pred_dic),
        'learning_rate': params['LEARNING_RATE'],
        'batch_size': params['BATCH_SIZE'],
        'add_layers': params['ADD_LAYERS'],
        'act_function':params['ACT_FUNCTION'],
        'drop_out_percent': params['DROP_OUT_PERCENT'],
        'margin': params['MARGIN']
    }

    # append word embedding related parameters to the dictionary
    if params['WORD_EMBEDDING']:
        er_mlp_params['num_entity_words'] = num_entity_words
        er_mlp_params['num_pred_words'] = num_pred_words
        er_mlp_params['indexed_entities'] = indexed_entities
        er_mlp_params['indexed_predicates'] = indexed_predicates

    #########################
    # construct the network #
    #########################
    er_mlp = ERMLP(er_mlp_params)

    # network used for training
    train_predictions = er_mlp.inference_for_max_margin_training()
    tf.add_to_collection('train_predictions', train_predictions)

    # network used for testing
    test_predictions = er_mlp.inference()
    tf.add_to_collection('test_predictions', test_predictions)

    # margin based ranking loss
    cost = er_mlp.loss()
    tf.add_to_collection('cost', cost)
    tf.summary.scalar('cost', cost)

    # optimizer
    if params['OPTIMIZER'] == 0:
        # adagrad
        optimizer = er_mlp.train_adagrad(cost)
    else:
        # adam
        optimizer = er_mlp.train_adam(cost)

    tf.add_to_collection('optimizer', optimizer)

    # merge summary
    merged = tf.summary.merge_all()

    # saver to save the model
    saver = tf.train.Saver()

    # choose the training data to actually train on
    data_train = indexed_train_data[indexed_train_data[:, 3] == 1]
    data_train = data_train[:, :3]

    # some variable initializations
    iter_list = []
    cost_list = []
    train_local_map_list = []
    test_map_list = []
    iteration = 0

    # init variables
    log.info('Initializing tensor variables...')
    init_all = tf.global_variables_initializer()

    log.info('Begin training...')
    # begin session
    with tf.Session() as sess:
        # writer
        train_writer = tf.summary.FileWriter(os.path.join(params['MODEL_SAVE_DIRECTORY'], 'log'), sess.graph)

        # run init
        sess.run(init_all)

        # epoch
        for epoch in range(params['TRAINING_EPOCHS']):
            log.info('****** Epoch: %d/%d ******', epoch, params['TRAINING_EPOCHS'])

            total_batch = int(np.ceil(data_train.shape[0] / params['BATCH_SIZE']))

            # shuffle the training data for each epoch
            np.random.shuffle(data_train)

            # iteration
            for i in range(total_batch):
                # get corrupted batch using the un-corrupted data_train
                start_idx = i*params['BATCH_SIZE']
                end_idx = (i+1)*params['BATCH_SIZE']
                batch_xs = er_mlp.get_training_batch_with_corrupted(data_train[start_idx:end_idx])

                # flip bit
                flip = bool(random.getrandbits(1))

                # feed dictionary
                feed_dict = {
                    er_mlp.train_triplets: batch_xs,
                    er_mlp.flip_placeholder: flip}

                # display progress
                if (i == 0) and (epoch % params['DISPLAY_STEP'] == 0):
                    _, train_summary, current_cost = sess.run([optimizer, merged, cost], feed_dict=feed_dict)
                    train_writer.add_summary(train_summary, iteration)

                    log.info('current cost: %f', current_cost)

                    train_local_map = er_mlp.test_model(sess, indexed_train_local_data, pred_dic, _type='train local')
                    train_local_map_list.append(train_local_map)

                    if not final_model:
                        thresholds = er_mlp.determine_threshold(sess, indexed_dev_data, f1=params['F1_FOR_THRESHOLD'])
                        test_map = er_mlp.test_model(sess, indexed_test_data, pred_dic, threshold=thresholds, _type='current test')
                        test_map_list.append(test_map)

                    iter_list.append(iteration)
                    cost_list.append(current_cost)
                else:
                    sess.run(optimizer, feed_dict=feed_dict)

                # update iteration
                iteration += 1

        # close writers
        train_writer.close()

        if not final_model:
            # do final threshold determination and testing model
            log.info('determine threshold for classification')
            thresholds = er_mlp.determine_threshold(sess, indexed_dev_data, f1=params['F1_FOR_THRESHOLD'])
            er_mlp.test_model(sess, indexed_test_data, pred_dic, threshold=thresholds, _type='final')

        # plot the cost graph
        plot_cost(iter_list, cost_list, params['MODEL_SAVE_DIRECTORY'])
        plot_map(iter_list, train_local_map_list, params['MODEL_SAVE_DIRECTORY'], filename='train_local_map.png')
        if not final_model:
            plot_map(iter_list, test_map_list, params['MODEL_SAVE_DIRECTORY'], filename='map.png')

        # save the model & parameters if prompted
        if params['SAVE_MODEL']:
            saver.save(sess, os.path.join(params['MODEL_SAVE_DIRECTORY'], 'model'))
            log.info('model saved in: %s', params['MODEL_SAVE_DIRECTORY'])

            save_object = {
                'entity_dic': entity_dic,
                'pred_dic': pred_dic
            }

            if not final_model:
                save_object['thresholds'] = thresholds

            if params['WORD_EMBEDDING']:
                save_object['indexed_entities'] = indexed_entities
                save_object['indexed_predicates'] = indexed_predicates
                save_object['num_pred_words'] = num_pred_words
                save_object['num_entity_words'] = num_entity_words

            with open(os.path.join(params['MODEL_SAVE_DIRECTORY'], 'params.pkl'), 'wb') as output:
                pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)
