"""
Filename: er_mlp_cross_entropy.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
	

To-do:
	1. machine_translate() passes separator. Fix it so that it does not.
"""

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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import random
from tensorflow.python import debug as tf_debug
from scipy import interp
import random
from data_processor import DataProcessor
from er_mlp import ERMLP
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats, plot_cost
from data_orchestrator_ce import DataOrchestrator

OVER_SAMPLE = True

def over_sample(data_X, data_Y, pred_dic):
	ret_X_list =[]
	ret_Y_list =[]

	for k,v in pred_dic.items():
		predicate_indices = np.where(data_X[:,1] == v)[0]

		if np.shape(predicate_indices)[0] != 0:
			X = data_X[predicate_indices]
			Y = data_Y[predicate_indices]
			s = X[:,0]
			s = s.reshape((np.shape(s)[0], 1))
			p = X[:,1]
			p = p.reshape((np.shape(p)[0], 1))
			o = X[:,2]
			o = o.reshape((np.shape(o)[0], 1))
			t = Y.reshape((np.shape(Y)[0], 1))

			combined_data = np.concatenate((s, p, o, t), axis=1)
			positive_samples = combined_data[combined_data[:, 3] == 1]
			negative_samples = combined_data[combined_data[:, 3] == 0]
			allIdx = np.array(range(0, np.shape(positive_samples)[0]))
			idx = np.random.choice(allIdx, size=np.shape(negative_samples)[0], replace=True)
			os_positive_samples = positive_samples[idx]
			combined_data = np.concatenate((os_positive_samples, negative_samples), axis=0)
			ret_X = combined_data[:, :3]
			ret_Y = combined_data[:, 3]
			print(np.shape(positive_samples))
			print(np.shape(os_positive_samples))
			print(np.shape(negative_samples))
			print(np.shape(combined_data))
			ret_X_list.append(ret_X)
			ret_Y_list.append(ret_Y)

	return np.column_stack((np.concatenate(ret_X_list, axis=0), np.concatenate(ret_Y_list, axis=0)))
	# return np.concatenate(ret_X_list,axis=0),np.concatenate(ret_Y_list,axis=0)

def run_model(
	WORD_EMBEDDING,
	DATA_TYPE,
	EMBEDDING_SIZE,
	LAYER_SIZE,
	TRAINING_EPOCHS,
	BATCH_SIZE,
	LEARNING_RATE,
	DISPLAY_STEP,
	CORRUPT_SIZE,
	LAMBDA,
	OPTIMIZER,
	ACT_FUNCTION,
	ADD_LAYERS,
	DROP_OUT_PERCENT,
	DATA_PATH,
	SAVE_MODEL=False,
	MODEL_SAVE_DIRECTORY=None,
	TRAIN_FILE='train.txt'):

	processor = DataProcessor()
	# load the data
	train_df = processor.load(DATA_PATH + TRAIN_FILE)
	test_df = processor.load(DATA_PATH + 'test.txt')
	dev_df = processor.load(DATA_PATH + 'dev.txt')
	# numerically represent the entities, predicates, and words
	# entity_dic = processor.create_entity_dic(df.as_matrix())
	# pred_dic = processor.create_relation_dic(df.as_matrix())
	print("machine translation...")
	indexed_entities, num_entity_words, entity_dic,indexed_predicates, indexed_pred_word_embeddings, pred_dic,num_pred_words, num_entity_words = None, None, None, None, None, None, None, None
	if WORD_EMBEDDING:
		indexed_entities, num_entity_words, entity_dic = processor.machine_translate_using_word(DATA_PATH + '/entities.txt', EMBEDDING_SIZE)
		indexed_predicates, num_pred_words, pred_dic = processor.machine_translate_using_word(DATA_PATH + '/relations.txt', EMBEDDING_SIZE)
	else:
		entity_dic = processor.machine_translate(DATA_PATH + '/entities.txt', EMBEDDING_SIZE, separator='#SPACE#|#COMMA#|#SEMICOLON#|\W+')
		pred_dic = processor.machine_translate(DATA_PATH + '/relations.txt', EMBEDDING_SIZE, separator='#SPACE#|#COMMA#|#SEMICOLON#|\W+')

	# numerically represent the data
	print("Index:")
	indexed_train_data = processor.create_indexed_triplets_test(train_df.as_matrix(), entity_dic, pred_dic )
	indexed_test_data = processor.create_indexed_triplets_test(test_df.as_matrix(), entity_dic, pred_dic )
	indexed_dev_data = processor.create_indexed_triplets_test(dev_df.as_matrix(), entity_dic, pred_dic )

	indexed_train_data[:, 3][indexed_train_data[:, 3] == -1] = 0
	indexed_test_data[:, 3][indexed_test_data[:, 3] == -1] = 0
	indexed_dev_data[:, 3][indexed_dev_data[:, 3] == -1] = 0
	print(" - index training complete")
	# skf = StratifiedKFold(n_splits=SPLITS, shuffle=True)

	NUM_ENTITIES = len(entity_dic)
	NUM_PREDS = len(pred_dic)

	er_mlp_params = {
		'word_embedding': WORD_EMBEDDING,
		'embedding_size': EMBEDDING_SIZE,
		'layer_size': LAYER_SIZE,
		'corrupt_size': CORRUPT_SIZE,
		'lambda': LAMBDA,
		'num_entities':NUM_ENTITIES,
		'num_preds':NUM_PREDS,
		'num_entity_words':num_entity_words,
		'num_pred_words':num_pred_words,
		'indexed_entities':indexed_entities,
		'indexed_predicates': indexed_predicates,
		'learning_rate':LEARNING_RATE,
		'batch_size': BATCH_SIZE,
		'add_layers': ADD_LAYERS,
		'act_function':ACT_FUNCTION,
		'drop_out_percent': DROP_OUT_PERCENT
	}

	data_train = indexed_train_data[:, :3]
	labels_train = indexed_train_data[:, 3]
	before_over_sampled_indexed_train_data = indexed_train_data[:, :]
	print('indexed_train_data')
	print(np.shape(indexed_train_data))
	labels_train = labels_train.reshape((np.shape(labels_train)[0], 1))
	Y_scores = np.zeros(labels_train.shape)
	# skf = StratifiedKFold(Y, n_folds=SPLITS, shuffle=True)
	# for trainIdx, testIdx in skf:
	# for trainIdx, testIdx in skf.split(X, Y):
		# data_train = X[trainIdx]
		# labels_train = Y[trainIdx]

	er_mlp = ERMLP(er_mlp_params)

	triplets, weights, biases, constants, y = er_mlp.create_tensor_terms()

	print('network for training')
	training_predictions = er_mlp.inference(triplets, weights, biases, constants, training=True)

	print('network for predictions')
	predictions = er_mlp.inference(triplets, weights, biases, constants)

	tf.add_to_collection('training_predictions', training_predictions)
	tf.add_to_collection('predictions', predictions)

	cost = er_mlp.loss_cross_entropy(training_predictions, y)

	tf.add_to_collection('cost', cost)

	print('optimizer')
	if OPTIMIZER == 0:
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

	if OVER_SAMPLE:
		indexed_train_data = over_sample(data_train,labels_train,pred_dic)
		# data_train,labels_train = over_sample(data_train,labels_train,pred_dic)

	def determine_threshold(indexed_data_dev, f1=True):
		# use the dev set to compute the best threshold for classification
		data_dev = indexed_data_dev[:, :3]
		predicates_dev = indexed_data_dev[:, 1]
		labels_dev = indexed_data_dev[:, 3]
		labels_dev = labels_dev.reshape((np.shape(labels_dev)[0], 1))
		predictions_dev = sess.run(predictions, feed_dict={triplets: data_dev, y: labels_dev})
		threshold = er_mlp.compute_threshold(predictions_dev, labels_dev, predicates_dev, f1)

		return threshold

	def test_model(indexed_train_data, indexed_data_test, threshold, _type='current'):
		data_train = indexed_train_data[:, :3]
		labels_train = indexed_train_data[:, 3]
		labels_train = labels_train.reshape((np.shape(labels_train)[0], 1))
		predicates_train = indexed_train_data[:, 1]
		predictions_list_train = sess.run(predictions, feed_dict={triplets: data_train, y: labels_train})
		mean_average_precision_train = pr_stats(NUM_PREDS, labels_train, predictions_list_train, predicates_train, pred_dic)
		roc_auc_train = roc_auc_stats(NUM_PREDS, labels_train, predictions_list_train, predicates_train, pred_dic)

		classifications_train = er_mlp.classify(predictions_list_train, threshold, predicates_train)
		classifications_train = np.array(classifications_train).astype(int)
		labels_train = labels_train.astype(int)
		fl_measure_train = f1_score(labels_train, classifications_train)
		accuracy_train = accuracy_score(labels_train, classifications_train)
		print(_type + " train mean average precision:" + str(mean_average_precision_train))
		print(_type + " train f1 measure:" + str(fl_measure_train))
		print(_type + " train accuracy:" + str(accuracy_train))
		print(_type + " train roc auc:" + str(roc_auc_train))

		data_test = indexed_data_test[:, :3]
		labels_test = indexed_data_test[:, 3]
		labels_test = labels_test.reshape((np.shape(labels_test)[0], 1))
		predicates_test = indexed_data_test[:, 1]
		predictions_list_test = sess.run(predictions, feed_dict={triplets: data_test, y: labels_test})
		mean_average_precision_test = pr_stats(NUM_PREDS, labels_test, predictions_list_test, predicates_test, pred_dic)
		roc_auc_test = roc_auc_stats(NUM_PREDS, labels_test, predictions_list_test, predicates_test, pred_dic)
		classifications_test = er_mlp.classify(predictions_list_test, threshold, predicates_test)
		classifications_test = np.array(classifications_test).astype(int)
		labels_test = labels_test.astype(int)
		fl_measure_test = f1_score(labels_test, classifications_test)
		accuracy_test = accuracy_score(labels_test, classifications_test)


		print(_type + " test mean average precision:" + str(mean_average_precision_test))
		print(_type + " test f1 measure:" + str(fl_measure_test))
		print(_type + " test accuracy:" + str(accuracy_test))
		print(_type + " test roc auc:" + str(roc_auc_test))

	data_orch = DataOrchestrator(indexed_train_data, shuffle=True)

	iter_list = []
	cost_list = []
	iteration = 0
	current_cost = 0.
	current_accuracy = None
	current_threshold = None

	print("Begin training...")

	# iterate through epochs
	for epoch in range(TRAINING_EPOCHS):
		avg_cost = 0.
		total_batch = int(data_train.shape[0] / BATCH_SIZE)
		for i in range(total_batch):
			# randidx = np.random.randint(int(data_train.shape[0]), size = BATCH_SIZE)
			train_batch = data_orch.get_next_training_batch(BATCH_SIZE)
			# batch_xs = data_train[randidx, :]
			# batch_ys = labels_train[randidx]
			# batch_ys = batch_ys.reshape((np.shape(batch_ys)[0],1))
			batch_xs = train_batch[:, :3]
			batch_ys = train_batch[:, 3]
			batch_ys = batch_ys.reshape((np.shape(batch_ys)[0], 1))

			_, current_cost= sess.run([optimizer, cost], feed_dict={triplets: batch_xs, y: batch_ys})

			avg_cost += current_cost / total_batch
			cost_list.append(current_cost)
			iter_list.append(iteration)

			iteration += 1

		# Display progress
		if epoch % DISPLAY_STEP == 0:
			thresholds = determine_threshold(indexed_dev_data)
			test_model(before_over_sampled_indexed_train_data, indexed_test_data, thresholds, _type='current')
			print ("Epoch: %03d/%03d cost: %.9f - current_cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost, current_cost))

		data_orch.reset_data_index()

	thresholds = determine_threshold(indexed_dev_data)
	test_model(before_over_sampled_indexed_train_data, indexed_test_data, thresholds, _type='final')
	plot_cost(iter_list, cost_list, MODEL_SAVE_DIRECTORY)

	if SAVE_MODEL:
		saver.save(sess, MODEL_SAVE_DIRECTORY + '/model')
		print('model saved in: '+ MODEL_SAVE_DIRECTORY)
		save_object = {
			'thresholds': thresholds,
			'entity_dic': entity_dic,
			'pred_dic': pred_dic
		}
		if WORD_EMBEDDING:
			save_object['indexed_entities'] = indexed_entities
			save_object['indexed_predicates'] = indexed_predicates
			save_object['num_pred_words'] = num_pred_words
			save_object['num_entity_words'] = num_entity_words

		with open(MODEL_SAVE_DIRECTORY + '/params.pkl', 'wb') as output:
			pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)

	# return accuracy, roc_auc_stats(labels_test,predictions_list,predicates_test, er_mlp_params)

if __name__ == "__main__":
	WORD_EMBEDDING = False
	DATA_TYPE = 'ecoli'
	EMBEDDING_SIZE = 60 # size of each embeddings
	LAYER_SIZE = 60 # number of columns in the first layer
	TRAINING_EPOCHS = 100
	BATCH_SIZE = 500
	LEARNING_RATE = 0.01
	DISPLAY_STEP = 1
	CORRUPT_SIZE = 10
	LAMBDA = 0.0001
	OPTIMIZER = 1
	ACT_FUNCTION = 0
	ADD_LAYERS=2
	DROP_OUT_PERCENT = 0.5
	DATA_PATH = '/Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/ecoli_for_param_opt_processed/p100/d1/'

	run_model(
		WORD_EMBEDDING,
		DATA_TYPE,
		EMBEDDING_SIZE,
		LAYER_SIZE,
		TRAINING_EPOCHS,
		BATCH_SIZE,
		LEARNING_RATE,
		DISPLAY_STEP,
		CORRUPT_SIZE,
		LAMBDA,
		OPTIMIZER,
		ACT_FUNCTION,
		ADD_LAYERS,
		DROP_OUT_PERCENT,
		DATA_PATH)
