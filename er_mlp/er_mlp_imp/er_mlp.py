"""
Filename: er_mlp.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
	Building blocks to be used for constructing ER MLP.

To-do:
	1. why use tf.ones for the B weights?
	2. combine inference_for_max_margin_training() and inference() into one function?
		or at least separate word embedding lookup part into separate function
	3. change names E, P, C, B, b to more meaningful names
	4. namescope for the model
	5. cross_margin inside compute_threshold() is not used. why?
"""

import math
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

class ERMLP:
	"""
	Class which consists of building blocks to build different types of ER MLPs.
	"""

	def __init__(self, params):
		"""
		Constructor for class ERMLP.

		Inputs:
			params: dictionary containing all the parameters
		"""
		self.params = params

	def create_tensor_terms(self):
		"""
		Create all the tensors required to construct the model.

		Returns:
			triplets: SPOs of size (batch_size, 3)
			weights: dictionary of all the weights of the network
			biases: dictionary of all the biases of the network
			constants: constants related to word embedding
			y: tuth value for the triplet used for evaluation of size (batch size, 1)
		"""
		# testing triplets: subject, predicate, object
		triplets = tf.placeholder(tf.int32, shape=(None, 3), name='triplets')

		# truth value for the triplet used for evaluation
		y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

		###########
		# weights #
		###########
		# E: word embeddings for all of the entities
		# P: word embeddings for all of the predicates
		# C: weight matrix for the first layer
		# B: weight matrix for the second layer
		if self.params['word_embedding']:
			E_vocab_size = self.params['num_entity_words']
			P_vocab_size = self.params['num_pred_words']
		else:
			E_vocab_size = self.params['num_entities']
			P_vocab_size = self.params['num_preds']

		weights_stddev = 1.0 / math.sqrt(self.params['embedding_size'])

		weights = {
			# initialize word embeddings
			'E': tf.Variable(tf.random_uniform([E_vocab_size, self.params['embedding_size']], -0.001, 0.001), name='E'),
			'P': tf.Variable(tf.random_uniform([P_vocab_size, self.params['embedding_size']], -0.001, 0.001), name='P'),
			# weights and biases of the network
			'C': tf.Variable(tf.truncated_normal([3*self.params['embedding_size'], self.params['layer_size']], stddev=weights_stddev), name='C'),
			'B': tf.Variable(tf.ones([self.params['layer_size'], 1]), name='B')
		}

		# add more layers if necessary
		if self.params['add_layers'] > 0:
			for i in range(1, self.params['add_layers'] + 1):
				weights['C{}'.format(i)] = tf.Variable(
					tf.truncated_normal([self.params['layer_size'], self.params['layer_size']], stddev=weights_stddev), name='C{}'.format(i))

		##########
		# biases #
		##########
		# bias for the first layer
		biases = {
			'b': tf.Variable(tf.zeros([1, self.params['layer_size']]), name='b')
		}

		# add more layers if necessary
		if self.params['add_layers'] > 0:
			for i in range(1, self.params['add_layers'] + 1):
				biases['b{}'.format(i)] = tf.Variable(tf.zeros([1, self.params['layer_size']]), name='b{}'.format(i))

		#############
		# constants #
		#############
		# gets constants required for word embeddings
		if self.params['word_embedding']:
			padded_entity_embedding_weights, padded_entity_indices =  self.get_padded_indices_and_weights(self.params['indexed_entities'])
			padded_predicate_embedding_weights, padded_predicate_indices =  self.get_padded_indices_and_weights(self.params['indexed_predicates'])

			constants = {
				'padded_entity_indices': tf.constant(padded_entity_indices, dtype=tf.int32),
				'padded_predicate_indices': tf.constant(padded_predicate_indices, dtype=tf.int32),
				'padded_entity_embedding_weights': tf.constant(padded_entity_embedding_weights, dtype=tf.float32),
				'padded_predicate_embedding_weights': tf.constant(padded_predicate_embedding_weights, dtype=tf.float32)
			}
		else:
			constants = None

		return triplets, weights, biases, constants, y

	def inference_for_max_margin_training(self, _triplets, _weights, _biases, _constants, _flip):
		"""
		Neural network that is used for training. Along with the evaluation of a triplet,
		the evaluation of a corrupted triplet is calculated as well so that we can calulcate
		the contrastive max margin loss.

		Inputs:
			_triplets: training triplets (subject, predicate, object, corrupted_entity)
			_weights: dictionary of all the weights of the network
			_biases: dictionary of all the biases of the network
			_constants: rest of the constants to be used in the network
			_flip: boolean to create a corrupt triplet by either corrupting
				the head or tail entity

		Returns:
			out: concatenation of correct output in 1st column and
				corrupted output in the 2nd column
		"""
		if self.params['word_embedding']:
			# look up indices in a list of embedding tensors and return
			# a tensor containing the embeddings (dense vectors) for each of the vocabularies
			# entity_embedded_word_ids.get_shape() = (8333, 2, 50)
			pred_embedded_word_ids = tf.nn.embedding_lookup(_weights['P'], _constants['padded_predicate_indices'])
			entity_embedded_word_ids = tf.nn.embedding_lookup(_weights['E'], _constants['padded_entity_indices'])

			# calculate weighted version of these embeddings
			# _constants['padded_entity_embedding_weights'].get_shape() = (8333, 2, 1)
			# entity_weighted_sum_of_embeddings.get_shape() = (8333, 2, 50)
			pred_weighted_sum_of_embeddings = tf.multiply(pred_embedded_word_ids, _constants['padded_predicate_embedding_weights'])
			entity_weighted_sum_of_embeddings = tf.multiply(entity_embedded_word_ids, _constants['padded_entity_embedding_weights'])

			# find sum of weighted embeddings calculated above
			# entity_emb.get_shape() = (8333, 50)
			pred_emb = tf.reduce_sum(pred_weighted_sum_of_embeddings, 1)
			entity_emb = tf.reduce_sum(entity_weighted_sum_of_embeddings, 1)
		else:
			pred_emb = _weights['P']
			entity_emb = _weights['E']

		# split the input
		sub, pred, obj, corrupt = tf.split(tf.cast(_triplets, tf.int32), 4, 1)

		# for each term of each sample, select the required embedding
		# and remove the extra dimension caused by the selection
		sub_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, sub))
		obj_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, obj))
		pred_emb = tf.squeeze(tf.nn.embedding_lookup(pred_emb, pred))
		corrupt_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, corrupt))
		sub_correct_emb = sub_emb
		obj_correct_emb = obj_emb

		# create a corrupt triplet, either corrupting the head or tail entity based on flip boolean
		sub_corrupted_emb, obj_corrupt_emb = tf.cond(_flip, lambda: (sub_emb, corrupt_emb), lambda: (corrupt_emb, obj_emb))

		# calculation of the first layer involves concatenating the three
		# embeddings for each sample and multipling it by the weight vector
		correct_pre_act = tf.add(tf.matmul(tf.concat([sub_correct_emb, pred_emb, obj_correct_emb], 1), _weights['C']), _biases['b'])
		corrupted_pre_act = tf.add(tf.matmul(tf.concat([sub_corrupted_emb, pred_emb, obj_corrupt_emb], 1), _weights['C']), _biases['b'])

		# add more layers if necessary
		if self.params['add_layers'] > 0:
			for i in range(1, self.params['add_layers'] + 1):
				correct_post_act = tf.nn.relu(correct_pre_act)
				corrupted_post_act = tf.nn.relu(corrupted_pre_act)

				correct_dropout = tf.nn.dropout(correct_post_act, self.params['drop_out_percent'])
				corrupted_dropout = tf.nn.dropout(corrupted_post_act, self.params['drop_out_percent'])

				correct_pre_act = tf.add(tf.matmul(correct_dropout, _weights['C'+str(i)]), _biases['b'+str(i)])
				corrupted_pre_act = tf.add(tf.matmul(corrupted_dropout, _weights['C'+str(i)]), _biases['b'+str(i)])

		if self.params['act_function'] == 0:
			# tanh
			print('using tanh for pre-final layer activation')
			pre_final_correct = tf.tanh(correct_pre_act)
			pre_final_corrupted = tf.tanh(corrupted_pre_act)
		else:
			# sigmoid
			print('using sigmoid for pre-final layer activation')
			pre_final_correct = tf.sigmoid(correct_pre_act)
			pre_final_corrupted = tf.sigmoid(corrupted_pre_act)

		out_correct_pre_act = tf.matmul(pre_final_correct, _weights['B'])
		out_corrupted_pre_act = tf.matmul(pre_final_corrupted, _weights['B'])

		out_correct = tf.sigmoid(out_correct_pre_act)
		out_corrupted = tf.sigmoid(out_corrupted_pre_act)

		out = tf.concat([out_correct, out_corrupted], axis=1, name='inference_for_max_margin_training')

		return out

	def inference(self, _triplets, _weights, _biases, _constants, training=False):
		"""
		Similar to the network used for training,
		but without evaluating the corrupted triplet.
		This is used for testing.

		Inputs:
			_triplets: training triplets (subject, predicate, object, corrupted_entity)
			_weights: dictionary of all the weights of the network
			_biases: dictionary of all the biases of the network
			_constants: rest of the constants to be used in the network
			training: boolean to control the dropout

		Returns:
			out: score for edge existence
		"""
		if self.params['word_embedding']:
			# look up indices in a list of embedding tensors and return
			# a tensor containing the embeddings (dense vectors) for each of the vocabularies
			# entity_embedded_word_ids.get_shape() = (8333, 2, 50)
			pred_embedded_word_ids = tf.nn.embedding_lookup(_weights['P'], _constants['padded_predicate_indices'])
			entity_embedded_word_ids = tf.nn.embedding_lookup(_weights['E'], _constants['padded_entity_indices'])

			# calculate weighted version of these embeddings
			# _constants['padded_entity_embedding_weights'].get_shape() = (8333, 2, 1)
			# entity_weighted_sum_of_embeddings.get_shape() = (8333, 2, 50)
			pred_weighted_sum_of_embeddings = tf.multiply(pred_embedded_word_ids, _constants['padded_predicate_embedding_weights'])
			entity_weighted_sum_of_embeddings = tf.multiply(entity_embedded_word_ids, _constants['padded_entity_embedding_weights'])

			# find sum of weighted embeddings calculated above
			# entity_emb.get_shape() = (8333, 50)
			pred_emb = tf.reduce_sum(pred_weighted_sum_of_embeddings, 1)
			entity_emb = tf.reduce_sum(entity_weighted_sum_of_embeddings, 1)
		else:
			pred_emb = _weights['P']
			entity_emb = _weights['E']

		# split the input. now there is no corrupted entity in the dataset
		sub, pred, obj = tf.split(tf.cast(_triplets, tf.int32), 3, 1)
		sub_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, sub))
		obj_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, obj))
		pred_emb = tf.squeeze(tf.nn.embedding_lookup(pred_emb, pred))

		# calculation of the first layer involves concatenating the three
		# embeddings for each sample and multipling it by the weight vector
		pre_act = tf.add(tf.matmul(tf.concat([sub_emb, pred_emb, obj_emb], axis=1), _weights['C']), _biases['b'])

		# add more layers if necessary
		if self.params['add_layers'] > 0:
			for i in range(1, self.params['add_layers'] + 1):
				post_act = tf.nn.relu(pre_act)

				if training:
					dropout = tf.nn.dropout(post_act, self.params['drop_out_percent'])
					pre_act = tf.add(tf.matmul(dropout, _weights['C'+str(i)]), _biases['b'+str(i)])
				else:
					pre_act = tf.add(tf.matmul(post_act, _weights['C'+str(i)]), _biases['b'+str(i)])

		if self.params['act_function'] == 0:
			# tanh
			print('using tanh for pre-final layer activation')
			pre_final = tf.tanh(pre_act)
		else:
			# sigmoid
			print('using sigmoid for pre-final layer activation')
			pre_final = tf.sigmoid(pre_act)

		out_pre_act = tf.matmul(pre_final, _weights['B'], name='inference')

		out = tf.sigmoid(out_pre_act)

		return out

	def compute_threshold(self, predictions_list, dev_labels, predicates, f1=False, cross_margin=False):
		"""
		Determine the best threshold to use for classification.

		Inputs:
			predictions_list: prediction found by running a feed-forward of the model
			dev_labels: ground truth label to be compared with predictions_list
			predicates: numpy array containing all the predicates within the dataset
			f1: True is using F1 score, False if using accuracy score
			cross_margin: 

		Returns:
			best_threshold: numpy array of same size as self.params['num_preds']
				which contains the best threshold for each predicate
		"""
		# inits
		best_threshold = np.zeros(self.params['num_preds'])

		# make sure to change label is -1 not 0
		dev_labels[:][dev_labels[:] == 0] = -1
		predictions_list[:][predictions_list[:] == 0] = -1

		# for each predicate
		for i in range(self.params['num_preds']):
			# find which lines (indeces) contain the predicate
			predicate_indices = np.where(predicates == i)[0]

			if np.shape(predicate_indices)[0] != 0:
				# among actual predictions, get those with predicate of interest
				predicate_predictions = np.reshape(predictions_list[predicate_indices], (-1, 1))
				# among gt labels, get those with predicate of interest
				predicate_labels = np.reshape(dev_labels[predicate_indices], (-1, 1))

				# stack prediction / gt label column-wise and sort along first column
				both = np.column_stack((predicate_predictions, predicate_labels))
				both = both[both[:, 0].argsort()]

				# get a flattened array after the sort
				predicate_predictions = both[:, 0].ravel()
				predicate_labels = both[:, 1].ravel()

				# init accuracy
				accuracies = np.zeros(np.shape(predicate_predictions))

				# for each triple using the predicate
				for j in range(np.shape(predicate_predictions)[0]):
					# using individual prediction as a score
					score = predicate_predictions[j]
					# find the predicted label for all predictions
					predictions = (predicate_predictions >= score)*2 - 1

					if f1:
						accuracy = f1_score(predicate_labels, predictions)
					else:
						accuracy = accuracy_score(predicate_labels, predictions)

					accuracies[j] = accuracy

				# find all the indices that has the best accuracy
				indices = np.argmax(accuracies)
				best_threshold[i] = np.mean(predicate_predictions[indices])

		return best_threshold

	def loss(self, predictions):
		"""
		Define loss to be optimized using margin based ranking loss.

		Inputs:
			predictions: concatenation of correct output in 1st column and
				corrupted output in the 2nd column

		Returns:
			margin based ranking loss
		"""
		batch_size = tf.constant(self.params['batch_size'], dtype=tf.float32)
		max_with_margin_sum = tf.div(tf.reduce_sum(tf.maximum(
			tf.add(tf.subtract(predictions[:, 1], predictions[:, 0]), self.params['margin']), 0)), batch_size)
		l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

		return tf.add(max_with_margin_sum, tf.multiply(self.params['lambda'], l2))

	def loss_cross_entropy(self, predictions, y):
		"""
		Define loss to be optimized using cross entropy.

		Inputs:
			predictions: predicted output
			y: correct output to be used as label

		Returns:
			cross entropy loss
		"""
		cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=y)
		l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

		return tf.add(tf.reduce_mean(cost), tf.multiply(self.params['lambda'], l2))

	def loss_weighted_cross_entropy(self, predictions, y):
		"""
		Define loss to be optimized using weighted cross entropy.

		Inputs:
			predictions: predicted output
			y: correct output to be used as label

		Returns:
			weighted cross entropy loss
		"""
		ratio = 6689.0 / (117340.0 + 6689.0)
		pos_weight = 1.0 / ratio

		cost = tf.nn.weighted_cross_entropy_with_logits(logits=predictions, targets=y, pos_weight=pos_weight)
		l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

		return tf.add(tf.reduce_mean(cost), tf.multiply((self.params['lambda'], l2)))

	def train_adam(self, loss):
		"""
		Use adam as optimizer.

		Inputs:
			loss: loss to optimize

		Returns:
			adam optimizer
		"""
		return tf.train.AdamOptimizer(learning_rate=self.params['learning_rate']).minimize(loss)

	def train_adagrad(self, loss):
		"""
		Use adagrad as optimizer.

		Inputs:
			loss: loss to optimize

		Returns:
			adagrad optimizer
		"""
		return tf.train.AdagradOptimizer(learning_rate=self.params['learning_rate']).minimize(loss)

	def classify(self, predictions_list, threshold, predicates, cross_margin=False):
		"""
		Using the best threshold computed for each predicate,
		perform classification on the predicted values.

		Inputs:
			predictions_list: prediction found by running a feed-forward of the model
			threshold: numpy array of same size as self.params['num_preds']
				which contains the best threshold for each predicate
			predicates: numpy array containing all the predicates within the dataset
			cross_margin: set to True if using cross margin

		Returns:
			classifications: list of length len(predictions_list)
				containing the label (1 or 0/-1) depending on whether
				if cross_margin is True or False
		"""
		classifications = []

		for i in range(len(predictions_list)):
			if(predictions_list[i][0] >= threshold[predicates[i]]):
				classifications.append(1)
			else:
				if cross_margin:
					classifications.append(-1)
				else:
					classifications.append(0)

		return classifications

	def get_training_batch_with_corrupted(self, data):
		"""
		Used during the calculation of the max margin loss.
		Each training batch will include the number of samples
		multiplied by the number of corruptions we want to make per sample.

		Inputs:
			data: training data

		Returns:
			batch: corrupted training batch data of
				len(batch) = data.shape[0] x CORRUPT_SIZE
		"""
		batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, self.params['num_entities']-1))\
			for i in range(data.shape[0]) for j in range(self.params['corrupt_size'])]

		return batch

	def get_padded_indices_and_weights(self, indices):
		"""
		Given indexed entities, return a padded weight and indice numpy array
		which are used to represent entity / relation as an average of
		their constituting word vectors.

		Inputs:
			indices: list of lists [[], [], []] where each list
				contains word index ids for a single entity / relation
				ex) indices = [[0], [1], [0, 2], [3]]

		Returns:
			weights: weights to use for averaging constituting word vectors
				ex) weights = [ [1, 0],
								[1, 0],
								[0.5, 0.5],
								[1, 0] ]
			padded_indices: vocabulary ids to be used for embedding lookup
				ex) padded_indices = [ [0, 0],
							   [1, 0],
							   [0, 2],
							   [3, 0] ]
		"""
		# find length of each list inside indices
		lens = np.array([len(indices[i]) for i in range(len(indices))])
		mask = np.arange(lens.max()) < lens[:, None]

		padded_indices = np.zeros(mask.shape, dtype=np.int32)
		padded_indices[mask] = np.hstack((indices[:]))

		weights = np.multiply(np.ones(mask.shape, dtype=np.float32) / lens[:, None], mask)
		weights = np.expand_dims(weights, self.params['embedding_size']-1)

		return weights, padded_indices
