"""
Filename: predict.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    

To-do:
"""
import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import model_global
from er_mlp import ERMLP
from data_processor import DataProcessor
from config_parser import ConfigParser

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate network.')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        default='./',
        help='Base directory')
    parser.add_argument(
        '--use_calibration',
        action='store_const',
        default=False,
        const=True)
    parser.add_argument(
        '--predict_file',
        required=True)
    parser.add_argument(
        '--logfile',
        default='',
        help='Path to save the log')

    return parser.parse_args()

def calibrate_probabilties(predictions_list_test, calibration_models, predicates_test, pred_dic, log_reg_calibrate):
    for _, i in pred_dic.items():
        indices, = np.where(predicates_test == i)

        if np.shape(indices)[0] != 0:
            predictions_predicate = predictions_list_test[indices]
            clf = calibration_models[i]

            if log_reg_calibrate:
                p_calibrated = clf.predict_proba(predictions_predicate.reshape(-1, 1))[:, 1]
            else:
                p_calibrated = clf.transform(predictions_predicate.ravel())

            predictions_list_test[indices] = np.reshape(p_calibrated, (-1, 1))

    return predictions_list_test

def main():
    """
    Main function.
    """
    # set log and parse args
    args = parse_argument()
    model_global.set_logging(args.logfile)

    # some init
    calibrated = args.use_calibration

    # directory and filename setup
    model_instance_dir = 'model_instance'
    model_save_dir = os.path.join(model_instance_dir, args.dir)
    config_file = os.path.join(model_save_dir, '{}.ini'.format(args.dir))

    # setup configuration parser
    configparser = ConfigParser(config_file)

    with tf.Session() as sess:
        # load the saved parameters
        with open(os.path.join(model_save_dir, 'params.pkl'), 'rb') as _file:
            params = pickle.load(_file)

        # some parameters
        entity_dic = params['entity_dic']
        pred_dic = params['pred_dic']
        thresholds = params['thresholds']

        if calibrated:
            calibration_models = params['calibrated_models']
            thresholds = params['thresholds_calibrated']

        er_mlp_params = {
            'word_embedding': configparser.getbool('WORD_EMBEDDING'),
            'embedding_size': configparser.getint('EMBEDDING_SIZE'),
            'layer_size': configparser.getint('LAYER_SIZE'),
            'corrupt_size': configparser.getint('CORRUPT_SIZE'),
            'lambda': configparser.getfloat('LAMBDA'),
            'num_entities': len(entity_dic),
            'num_preds': len(pred_dic),
            'learning_rate': configparser.getfloat('LEARNING_RATE'),
            'batch_size': configparser.getint('BATCH_SIZE'),
            'add_layers': configparser.getint('ADD_LAYERS'),
            'act_function': configparser.getint('ACT_FUNCTION'),
            'drop_out_percent': configparser.getfloat('DROP_OUT_PERCENT')
        }

        if configparser.getbool('WORD_EMBEDDING'):
            num_entity_words = params['num_entity_words']
            num_pred_words = params['num_pred_words']
            indexed_entities = params['indexed_entities']
            indexed_predicates = params['indexed_predicates']
            er_mlp_params['num_entity_words'] = num_entity_words
            er_mlp_params['num_pred_words'] = num_pred_words
            er_mlp_params['indexed_entities'] = indexed_entities
            er_mlp_params['indexed_predicates'] = indexed_predicates

        # init ERMLP class using the parameters defined above
        er_mlp = ERMLP(
            er_mlp_params,
            sess,
            meta_graph=os.path.join(model_save_dir, 'model.meta'),
            model_restore=os.path.join(model_save_dir, 'model'))

        processor = DataProcessor()
        test_df = processor.load(os.path.join(configparser.getstr('DATA_PATH'), args.predict_file))

        if test_df.shape[1] == 4:
            indexed_data_test = processor.create_indexed_triplets_test(
                test_df.values, entity_dic, pred_dic)
        else:
            indexed_data_test = processor.create_indexed_triplets_training(
                test_df.values, entity_dic, pred_dic)

        data_test = indexed_data_test[:, :3]
        predicates_test = indexed_data_test[:, 1]

        predictions_list_test = sess.run(
            er_mlp.test_predictions, feed_dict={er_mlp.test_triplets: data_test})

        if calibrated:
            predictions_list_test = calibrate_probabilties(
                predictions_list_test, calibration_models, predicates_test,
                pred_dic, configparser.getbool('LOG_REG_CALIBRATE'))

        classifications_test = er_mlp.classify(predictions_list_test, thresholds, predicates_test)
        classifications_test = np.array(classifications_test).astype(int)
        classifications_test = classifications_test.reshape((np.shape(classifications_test)[0], 1))

        c = np.dstack((classifications_test, predictions_list_test))
        c = np.squeeze(c)

        if test_df.shape[1] == 4:
            labels_test = indexed_data_test[:, 3]
            labels_test = labels_test.reshape((np.shape(labels_test)[0], 1))

            c = np.concatenate((c, labels_test), axis=1)

        predict_folder = os.path.splitext(args.predict_file)[0]
        predict_folder = os.path.join(model_save_dir, predict_folder)

        if not os.path.exists(predict_folder):
            os.makedirs(predict_folder)

        with open(os.path.join(predict_folder, 'predictions.txt'), 'w') as _file:
            for i in range(np.shape(c)[0]):
                if test_df.shape[1] == 4:
                    _file.write('predicate: ' + str(predicates_test[i]) + '\tclassification: ' + str(int(c[i][0])) + '\tprediction: ' + str(c[i][1]) + '\tlabel: ' + str(int(c[i][2])) + '\n')
                else:
                    _file.write('predicate: ' + str(predicates_test[i]) + '\tclassification: ' + str(int(c[i][0])) + '\tprediction: ' + str(c[i][1]) + '\n')

if __name__ == '__main__':
    main()
