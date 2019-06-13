"""
Filename: determine_thresholds.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Determine the threshold.

To-do:
    1. why set label to 0 now instead of -1?
"""
import os
import argparse
import logging as log
import pickle
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
    parser = argparse.ArgumentParser(description='Determine the thresholds.')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        default='./',
        help='Base directory')
    parser.add_argument(
        '--logfile',
        default='',
        help='Path to save the log')

    return parser.parse_args()

def main():
    """
    Main function.
    """
    # set log and parse args
    args = parse_argument()
    model_global.set_logging(args.logfile)

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

        log.info('Original thresholds: %s', params['thresholds'])

        # some parameters
        entity_dic = params['entity_dic']
        pred_dic = params['pred_dic']

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
            'act_function':configparser.getint('ACT_FUNCTION'),
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
        dev_df = processor.load(os.path.join(configparser.getstr('DATA_PATH'), 'dev.txt'))
        indexed_data_dev = processor.create_indexed_triplets_test(dev_df.values, entity_dic, pred_dic)
        indexed_data_dev[:, 3][indexed_data_dev[:, 3] == -1] = 0

        # find the threshold
        thresholds = er_mlp.determine_threshold(
            sess, indexed_data_dev, f1=configparser.getbool('F1_FOR_THRESHOLD'))

        log.debug('thresholds: %s', thresholds)

        # define what to save
        save_object = {
            'entity_dic': entity_dic,
            'pred_dic': pred_dic,
            'thresholds': thresholds
        }

        if hasattr(params, 'thresholds_calibrated'):
            save_object['thresholds_calibrated'] = params['thresholds_calibrated']

        if hasattr(params, 'calibrated_models'):
            save_object['calibrated_models'] = params['calibrated_models']

        if configparser.getbool('WORD_EMBEDDING'):
            save_object['indexed_entities'] = indexed_entities
            save_object['indexed_predicates'] = indexed_predicates
            save_object['num_pred_words'] = num_pred_words
            save_object['num_entity_words'] = num_entity_words

        # save the parameters
        with open(os.path.join(model_save_dir, 'params.pkl'), 'wb') as output:
            pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
