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
"""

import os
import sys
import random
import numpy as np
import pickle as pickle
import tensorflow as tf
from er_mlp import ERMLP
from data_processor import DataProcessor
from metrics import roc_auc_stats, pr_stats, plot_cost
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def run_model(params):
	"""
	Run the ER_MLP model using max margin loss.

	Inputs:
		params: dictionary containing different
			parameters to be used when running the model
	"""
	# numerically represent the entities, predicates, and words
	processor = DataProcessor()

	# load train / dev / test data as dataframe
	train_df = processor.load(params['DATA_PATH'] + params['TRAIN_FILE'])
	dev_df = processor.load(params['DATA_PATH'] + 'dev.txt')
	test_df = processor.load(params['DATA_PATH'] + 'test.txt')

	print('train dataframe shape: {}'.format(train_df.shape))
	print('dev dataframe shape: {}'.format(dev_df.shape))
	print('test dataframe shape: {}'.format(test_df.shape))

	if len(train_df.columns) < 4:
		train_df['one'] = 1

	print('performing machine translation...')
	if params['WORD_EMBEDDING']:
		indexed_entities, num_entity_words, entity_dic = processor.machine_translate_using_word(params['DATA_PATH'] + '/entities.txt')
		indexed_predicates, num_pred_words, pred_dic = processor.machine_translate_using_word(params['DATA_PATH'] + '/relations.txt')
	else:
		entity_dic = processor.machine_translate(params['DATA_PATH'] + '/entities.txt')
		pred_dic = processor.machine_translate(params['DATA_PATH'] + '/relations.txt')

	# numerically represent the data
	indexed_train_data = processor.create_indexed_triplets_test(train_df.as_matrix(), entity_dic, pred_dic)
	indexed_dev_data = processor.create_indexed_triplets_test(dev_df.as_matrix(), entity_dic, pred_dic)
	indexed_test_data = processor.create_indexed_triplets_test(test_df.as_matrix(), entity_dic, pred_dic)

	# change label from 0 to -1 for test / dev data
	indexed_dev_data[:, 3][indexed_dev_data[:, 3] == 0] = -1
	indexed_test_data[:, 3][indexed_test_data[:, 3] == 0] = -1

	# shuffle test data
	np.random.shuffle(indexed_test_data)

	# find number of entities and predicates
	NUM_ENTITIES = len(entity_dic)
	NUM_PREDS = len(pred_dic)

	# construct new parameter dictionary to be actually fed into the network
	er_mlp_params = {
		'word_embedding': params['WORD_EMBEDDING'],
		'embedding_size': params['EMBEDDING_SIZE'],
		'layer_size': params['LAYER_SIZE'],
		'corrupt_size': params['CORRUPT_SIZE'],
		'lambda': params['LAMBDA'],
		'num_entities': NUM_ENTITIES,
		'num_preds': NUM_PREDS,
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

	# create tensors for model construction
	triplets, weights, biases, constants, y = er_mlp.create_tensor_terms()

	# training triplets: subject, predicate, object, corrupted_entity
	training_triplets = tf.placeholder(tf.int32, shape=(None, 4), name='training_triplets')

	# boolean to determine if we want to corrupt the head or tail
	flip_placeholder = tf.placeholder(tf.bool, name='flip_placeholder')

	# network used for training
	training_predictions = er_mlp.inference_for_max_margin_training(training_triplets, weights, biases, constants, flip_placeholder)
	tf.add_to_collection('training_predictions', training_predictions)

	# network used for testing
	predictions = er_mlp.inference(triplets, weights, biases, constants)
	tf.add_to_collection('predictions', predictions)

	# margin based ranking loss
	cost = er_mlp.loss(training_predictions)
	tf.add_to_collection('cost', cost)

	# optimizer
	if params['OPTIMIZER'] == 0:
		# adagrad
		optimizer = er_mlp.train_adagrad(cost)
	else:
		# adam
		optimizer = er_mlp.train_adam(cost)

	tf.add_to_collection('optimizer', optimizer)

	# init variables
	print('initializing tensor variables...')
	init_all = tf.global_variables_initializer()

	# begin session
	sess = tf.Session()

	# saver to save the model
	saver = tf.train.Saver()

	# run session
	sess.run(init_all)

	# choose the training data to actually train on
	data_train = indexed_train_data[indexed_train_data[:, 3] == 1]
	data_train = data_train[:, :3]

	def _determine_threshold(indexed_data_dev, f1=False):
		"""
		Use the dev set to compute the best thresholds for classification.

		Inputs:
			indexed_data_dev: development data set
			f1: True is using F1 score, False if using accuracy score

		Returns:
			threshold: numpy array of same size as self.params['num_preds']
				which contains the best threshold for each predicate
		"""
		data_dev = indexed_data_dev[:, :3]
		predicates_dev = indexed_data_dev[:, 1]
		labels_dev = np.reshape(indexed_data_dev[:, 3], (-1, 1))

		predictions_dev = sess.run(predictions, feed_dict={triplets: data_dev, y: labels_dev})

		threshold = er_mlp.compute_threshold(predictions_dev, labels_dev, predicates_dev, f1, cross_margin=True)

		return threshold

	def _test_model(indexed_data_test, threshold, _type='current'):
		"""
		Test the model using the optimal threshold found.

		Inputs:
			indexed_data_test: test data set
			threshold: numpy array of same size as self.params['num_preds']
				which contains the best threshold for each predicate
		"""
		data_test = indexed_data_test[:, :3]
		predicates_test = indexed_data_test[:, 1]
		labels_test = np.reshape(indexed_data_test[:, 3], (-1, 1))

		predictions_test = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})

		# find mAP and auc
		mean_average_precision_test = pr_stats(NUM_PREDS, labels_test, predictions_test, predicates_test, pred_dic)
		roc_auc_test = roc_auc_stats(NUM_PREDS, labels_test, predictions_test, predicates_test, pred_dic)

		# get test classification
		classifications_test = er_mlp.classify(predictions_test, threshold, predicates_test, cross_margin=True)
		classifications_test = np.array(classifications_test).astype(int)

		# get confusion matrix
		confusion_test = confusion_matrix(labels_test, classifications_test)

		# find F1 & accuracy
		labels_test = labels_test.astype(int)
		fl_measure_test = f1_score(labels_test, classifications_test)
		accuracy_test = accuracy_score(labels_test, classifications_test)

		# for each predicate
		for i in range(NUM_PREDS):
			# find the corresponding string for the predicate
			for key, value in pred_dic.items():
				if value == i:
					pred_name = key

			# find which lines (indeces) contain the predicate
			indices = np.where(predicates_test == i)[0]

			if np.shape(indices)[0] != 0:
				classifications_predicate = classifications_test[indices]
				labels_predicate = labels_test[indices]

				# find F1, accuracy, and confusion matrix
				fl_measure_predicate = f1_score(labels_predicate, classifications_predicate)
				accuracy_predicate = accuracy_score(labels_predicate, classifications_predicate)
				confusion_predicate = confusion_matrix(labels_predicate, classifications_predicate)

				print(' - test f1 measure for {}: {}'.format(pred_name, fl_measure_predicate))
				print(' - test accuracy for {}: {}'.format(pred_name, accuracy_predicate))
				print(' - test confusion matrix for {}: '.format(pred_name))
				print(confusion_predicate)
				print(' ')

		# print stats for the whole dataset
		print('{} test mean average precision: {}'.format(_type, mean_average_precision_test))
		print('{} test f1 measure: {}'.format(_type, fl_measure_test))
		print('{} test accuracy: {}'.format(_type, accuracy_test))
		print('{} test roc auc: {}'.format(_type, roc_auc_test))
		print('{} test confusion matrix: '.format(_type))
		print(confusion_test)
		print(' ')
		print(' ')
		print(' ')

	# some variable initializations
	iter_list = []
	cost_list = []
	iteration = 0
	current_cost = 0.

	print('Begin training...')

	# epoch
	for epoch in range(params['TRAINING_EPOCHS']):
		avg_cost = 0.
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

			_, current_cost = sess.run([optimizer, cost], feed_dict={training_triplets: batch_xs, flip_placeholder: flip})

			avg_cost += current_cost / total_batch
			iter_list.append(iteration)
			cost_list.append(current_cost)

			# update iteration
			iteration += 1

		# display progress
		if epoch % params['DISPLAY_STEP'] == 0:
			thresholds = _determine_threshold(indexed_dev_data, f1=params['F1_FOR_THRESHOLD'])
			_test_model(indexed_test_data, thresholds, _type='current')
			print('Epoch: %03d/%03d cost: %.9f - current_cost: %.9f' % (epoch, params['TRAINING_EPOCHS'], avg_cost, current_cost))
			print('')

	# do final threshold determination and testing model
	print('determine threshold for classification')
	thresholds = _determine_threshold(indexed_dev_data, f1=params['F1_FOR_THRESHOLD'])
	_test_model(indexed_test_data, thresholds, _type='final')

	# plot the cost graph
	plot_cost(iter_list, cost_list, params['MODEL_SAVE_DIRECTORY'])

	# save the model & parameters if prompted
	if params['SAVE_MODEL']:
		saver.save(sess,params['MODEL_SAVE_DIRECTORY'] + '/model')
		print('model saved in: ' + params['MODEL_SAVE_DIRECTORY'])

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

		with open(params['MODEL_SAVE_DIRECTORY'] + '/params.pkl', 'wb') as output:
			pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)
