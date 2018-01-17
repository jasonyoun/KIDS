import numpy as np
import tensorflow as tf
import random
import math

class ERMLP:
    def __init__(self,params):
        self.params = params

    def create_tensor_terms(self):
        # The training triplets: subject, predicate, object, corrupted_entity
        # self.training_triplets = tf.placeholder(tf.int32, shape=(None, 4))

        # The testing triplets: subject, predicate, object
        triplets = tf.placeholder(tf.int32, shape=(None, 3))

        # The tuth value for the triplet used for evaluation
        y = tf.placeholder(tf.float32, [None, 1])

        # A boolean to determine if we want to corrupt the head or tail
        # self.flip_placeholder = tf.placeholder(tf.bool)

        # the weights:
        #   E: the word embeddings for all of the entities
        #   P: The word embeddings for all of the predicates
        #   C: The Weight matrix for the first layer
        #   B: The weight matrix for the second layer
        E_size = None
        P_size = None
        if self.params['word_embedding']:
            E_size = self.params['num_entity_words']
            P_size = self.params['num_pred_words']
        else:
            E_size = self.params['num_entities']
            P_size = self.params['num_preds']

        print("weights")
        weights = {
            'E': tf.Variable(tf.random_uniform([E_size,self.params['embedding_size']], -0.001, 0.001), name='E'),
            'P': tf.Variable(tf.random_uniform([P_size,self.params['embedding_size']], -0.001, 0.001), name='P'),
            'C': tf.Variable(tf.truncated_normal([3*self.params['embedding_size'],self.params['layer_size']],stddev=1.0 / math.sqrt(self.params['embedding_size'])), name='C'),
            'B': tf.Variable(tf.ones([self.params['layer_size'],1]))
        }

        # The bias for the first layer
        print("biases")
        biases = {
            'b': tf.Variable(tf.zeros([1,self.params['layer_size']]),name='b')#,    

        # incase we need an output bias
        #    'out': tf.Variable(tf.random_normal([1]))
        }

        # The constants:
        constants = None
        if self.params['word_embedding']:
            padded_entity_embedding_weights,padded_entity_indices =  self.pad_indices_with_weights(self.params['indexed_entities'])
            padded_predicate_embedding_weights,padded_predicate_indices =  self.pad_indices_with_weights(self.params['indexed_predicates'])
            print("constants")
            constants = {
                'padded_entity_indices': tf.constant(padded_entity_indices,dtype=tf.int32),
                'padded_predicate_indices': tf.constant(padded_predicate_indices,dtype=tf.int32),
                'padded_entity_embedding_weights': tf.constant(padded_entity_embedding_weights,dtype=tf.float32),
                'padded_predicate_embedding_weights': tf.constant(padded_predicate_embedding_weights,dtype=tf.float32)
            }
        return triplets, weights, biases, constants, y

    # the neural network that is used during training. Along with the evaluation of a triplet,
    # the evaluation of a corrupted triplet is calculated as well so that we can calulcate
    # the contrastive max margin loss
    def inference_for_max_margin_training(self,_triplets, _weights, _biases, _constants, _flip, _act_function):
        pred_emb = None
        entity_emb = None
        if self.params['word_embedding']:
            # create entity embeddings from words, padded indices, and weights
            print('pred embeddings')
            pred_emb = tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(_weights['P'],_constants['padded_predicate_indices']),_constants['padded_predicate_embedding_weights']),1)
            print('entity embeddings')
            entity_emb = tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(_weights['E'],_constants['padded_entity_indices']),_constants['padded_entity_embedding_weights']),1)
        else:
            pred_emb= _weights['P']
            entity_emb= _weights['E'] 
        
        # split the input
        sub, pred, obj, corrupt = tf.split( tf.cast(_triplets, tf.int32),4,1)
        # for each term of each sample, select the required embedding and remove the 
        # extra dimension caused by the selection
        sub_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, sub))

        obj_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, obj))
        pred_emb = tf.squeeze(tf.nn.embedding_lookup(pred_emb, pred))
        corrupt_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, corrupt))
        sub_correct_emb = sub_emb
        obj_correct_emb = obj_emb

        # create a corrupt triplet, either corrupting the head or tail entity based on flip boolean
        sub_corrupted_emb,obj_corrupt_emb = tf.cond(_flip, lambda: (sub_emb,corrupt_emb), lambda: (corrupt_emb,obj_emb))
        
        # calculation of the first layer involves concatenating the three embeddings for each sample
        # and multipling it by the weight vector 
        layer_1_correct = None
        layer_1_corrupted = None
        print('activation function:')
        if _act_function==0:
            print(str(_act_function)+ 'tanh')
            layer_1_correct = tf.tanh(tf.add(tf.matmul(tf.concat( [sub_correct_emb,pred_emb,obj_correct_emb],1),_weights['C']),_biases['b']))
            layer_1_corrupted = tf.tanh(tf.add(tf.matmul(tf.concat( [sub_corrupted_emb,pred_emb,obj_corrupt_emb],1),_weights['C']),_biases['b']))
        else:
            print(str(_act_function)+ 'sigmoid')
            layer_1_correct = tf.sigmoid(tf.add(tf.matmul(tf.concat( [sub_correct_emb,pred_emb,obj_correct_emb],1),_weights['C']),_biases['b']))
            layer_1_corrupted = tf.sigmoid(tf.add(tf.matmul(tf.concat( [sub_corrupted_emb,pred_emb,obj_corrupt_emb],1),_weights['C']),_biases['b']))

        #out = tf.add(tf.matmul(tf.transpose(_weights['B']),layer_1),biases['out'])
        out_correct = tf.matmul(layer_1_correct,_weights['B'])
        out_corrupted = tf.matmul(layer_1_corrupted,_weights['B'])
        out = tf.concat([out_correct, out_corrupted],axis=1)
        #out = tf.stack([out_correct, out_corrupted])
        return out

    # Similar to the network used for training, but without evaluating the corrupted triplet. This is used during testing
    def inference(self,_triplets, _weights, _biases, _constants,_act_function):
        pred_emb = None
        entity_emb = None
        if self.params['word_embedding']:
            # create entity embeddings from words, padded indices, and weights
            print('pred embeddings')
            pred_emb = tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(_weights['P'],_constants['padded_predicate_indices']),_constants['padded_predicate_embedding_weights']),1)
            print('entity embeddings')
            entity_emb = tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(_weights['E'],_constants['padded_entity_indices']),_constants['padded_entity_embedding_weights']),1)

        else:
            pred_emb= _weights['P']
            entity_emb= _weights['E'] 
        
        # split the input, now there is no corrupted entity in the dataset
        sub, pred, obj = tf.split( tf.cast(_triplets, tf.int32),3,1)
        sub_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, sub))
        obj_emb = tf.squeeze(tf.nn.embedding_lookup(entity_emb, obj))
        pred_emb = tf.squeeze(tf.nn.embedding_lookup(pred_emb, pred))
        
        # calculation of the first layer involves concatenating the three embeddings for each sample
        # and multipling it by the weight vector 
        layer_1 = None
        print('activation function:')
        if _act_function==0:
            print(str(_act_function)+ 'tanh')
            layer_1 = tf.tanh(tf.add(tf.matmul(tf.concat( [sub_emb,pred_emb,obj_emb],axis=1),_weights['C']),_biases['b']))
        else:
            print(str(_act_function)+ 'sigmoid')
            layer_1 = tf.sigmoid(tf.add(tf.matmul(tf.concat( [sub_emb,pred_emb,obj_emb],axis=1),_weights['C']),_biases['b']))
        #out = tf.add(tf.matmul(tf.transpose(_weights['B']),layer_1),biases['out'])
        out = tf.matmul(layer_1,_weights['B'])
        return out

    # determine the best threshold to use for classification
    def compute_threshold(self, predictions_list, dev_labels,predicates):
        min_score = np.min(predictions_list) 
        max_score = np.max(predictions_list) 
        best_threshold = np.zeros(self.params['num_preds']);
        best_accuracy = np.zeros(self.params['num_preds']);
        for i in range(self.params['num_preds']):
            best_threshold[i]= min_score;
            best_accuracy[i] = -1;

        score = min_score
        increment = 0.01
        while(score <= max_score):
            for i in range(self.params['num_preds']):
                predicate_indices = np.where(predicates == i)[0]
                predicate_predictions = predictions_list[predicate_indices]
                predictions = (predicate_predictions >= score) * 2 -1
                predicate_labels = dev_labels[predicate_indices]
                accuracy = np.mean(predictions == predicate_labels)
                if accuracy > best_accuracy[i]:
                    best_threshold[i] = score
                    best_accuracy[i] = accuracy
                score += increment
        return best_threshold

    def loss(self, predictions): 
        batch_size = tf.constant(self.params['batch_size'],dtype=tf.float32)
        one = tf.constant(1,dtype=tf.float32)
        max_with_margin_sum =tf.div(tf.reduce_sum(tf.maximum(tf.add(tf.subtract(predictions[:,1],predictions[:, 0]),1), 0)), batch_size)
        l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        return max_with_margin_sum + (self.params['lambda'] * l2)

    def loss_cross_entropy(self,predictions, y):  
        l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=y)
        return tf.reduce_mean(cost) + (self.params['lambda'] *  l2)

    def loss_weighted_cross_entropy(self,predictions, y):  
        ratio = 6689.0/ (117340.0 + 6689.0)
        pos_weight = 1.0 / ratio
        l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=predictions, targets=y, pos_weight=pos_weight)
        return tf.reduce_mean(cost)+ (self.params['lambda'] * l2)

    def train_adam(self,loss):
        return tf.train.AdamOptimizer(learning_rate = self.params['learning_rate']).minimize(loss)

    def train_adagrad(self,loss):
        return tf.train.AdagradOptimizer(learning_rate = self.params['learning_rate']).minimize(loss)

    def classify(self, predictions_list,threshold, predicates):
        classifications = []
        for i in range(len(predictions_list)):
            if(predictions_list[i][0] >= threshold[predicates[i]]):
                classifications.append(1)
            else:
                classifications.append(-1)
        return classifications

    # each training batch will include the number of samples multiplied by the number 
    # of corruptions we want to make per sample - used during the calculation of the 
    # max margin loss
    def get_training_batch_with_corrupted(self, data):
        random_indices = random.sample(range(len(data)), self.params['batch_size'])
        batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, self.params['num_entities']-1))\
            for i in random_indices for j in range(self.params['corrupt_size'])]
        return batch

    def pad_indices_with_weights(self,indices):
        lens = np.array([len(indices[i]) for i in range(len(indices))])
        mask = np.arange(lens.max()) < lens[:,None]
        padded = np.zeros(mask.shape, dtype=np.int32)
        padded[mask] = np.hstack((indices[:]))
        weights = np.multiply(np.ones(mask.shape,dtype=np.float32) / lens[:,None],mask)
        weights=np.expand_dims(weights,self.params['embedding_size']-1)
        return weights, padded
