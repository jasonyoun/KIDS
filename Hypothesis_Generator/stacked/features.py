"""
Filename: features.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Read the model predictions and parser them.

To-do:
"""

# standard imports
import os
import pickle

# third party imports
import numpy as np

ER_MLP_MODEL_HOME = '../er_mlp/model/model_instance/'
PRA_MODEL_HOME = '../pra/model/model_instance/'

def get_x_y(which, er_mlp, pra, final_model=False):
    """
    Load the model predictions information.

    Inputs:
        which: which data are we accessing (train_local, dev, etc.)
        er_mlp: directory of the er_mlp model
        pra: directory of the pra model

    Returns:
        pred_dic: dictionary where key is a predicate and value is the
            index assigned to that specific predicate
        x: numpy array where
            x[:, 0] = er_mlp prediction raw output
            x[:, 1] = pra prediction raw output
            x[:, 2] = valid / invalid depending on pra
        y: numpy array containing the ground truth label
        predicates: numpy array containing which predicate
            the (x, y) pair belong to using the indexing from pred_dic
    """
    model_base_dir = os.path.join(ER_MLP_MODEL_HOME, er_mlp)
    fn = open(os.path.join(model_base_dir, 'params.pkl'), 'rb')
    params = pickle.load(fn)
    pred_dic = params['pred_dic']

    ##########
    # er_mlp #
    ##########
    # open predictions and process it
    with open(os.path.join(os.path.join(model_base_dir, which), 'predictions.txt'), "r") as _file:
        predictions = _file.readlines()
    predictions = [x.strip() for x in predictions]

    labels = []
    er_mlp_features = []

    for line in predictions:
        strings = line.split('\t')
        predicate = int(strings[0].replace('predicate: ', ''))
        if not final_model:
            pred = float(strings[2].replace('prediction: ', ''))
            label = int(strings[3].replace('label: ', ''))
        else:
            pred = float(strings[1].replace('prediction: ', ''))
            label = int(strings[2].replace('label: ', ''))

        labels.append([label])
        er_mlp_features.append([predicate, pred, 1]) # all valid

    # convert to numpy arrays
    labels_array = np.array(labels)
    er_mlp_features_array = np.array(er_mlp_features)

    if len(er_mlp_features_array) > 1:
        e_features = np.vstack(er_mlp_features_array)
    else:
        e_features = er_mlp_features_array

    #######
    # pra #
    #######
    pra_features_array_list = []
    model_base_dir = os.path.join(os.path.join(PRA_MODEL_HOME, pra), 'instance')

    pra_features = []

    for key, val in pred_dic.items():
        scores_file = os.path.join(model_base_dir, which)
        scores_file = os.path.join(scores_file, 'scores')
        scores_file = os.path.join(scores_file, key)

        if os.path.isfile(scores_file):
            with open(scores_file, "r") as _file:
                predictions = _file.readlines()
                predictions = [x.strip() for x in predictions]

                for line in predictions:
                    strings = line.split('\t')
                    pred = float(strings[0].strip())
                    valid = int(strings[1].strip())
                    pra_features.append([int(val), pred, valid])

    pra_features_array = np.array(pra_features)
    pra_features_array_list.append(pra_features_array)

    if len(pra_features_array_list) > 1:
        p_features = np.vstack(pra_features_array_list)
    else:
        p_features = pra_features_array_list

    p_features = np.squeeze(p_features)

    ##################################
    # process the extracted features #
    ##################################
    predicates_er_mlp = e_features[:, 0]
    predicates_er_mlp = predicates_er_mlp.astype(int)

    predicates_pra = p_features[:, 0]
    predicates_pra = predicates_pra.astype(int)

    labels_list = []
    combined_list = []
    predicates_list = []

    for key, val in pred_dic.items():
        p_predicate_indices = np.where(predicates_pra[:] == val)[0]
        e_predicate_indices = np.where(predicates_er_mlp[:] == val)[0]
        labels_list.append(labels_array[e_predicate_indices])
        predicates_list.append(predicates_er_mlp[e_predicate_indices])
        combined_list.append(np.hstack((e_features[e_predicate_indices][:, 1:], p_features[p_predicate_indices][:, 1:])))

    y = np.vstack(labels_list)
    y[:][y[:] == -1] = 0
    predicates = np.hstack(predicates_list)
    combined_array = np.vstack(combined_list)
    x = combined_array[:, [0, 2, 3]]

    return pred_dic, x, y, predicates
