"""
Filename: build_network.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Build and run the ER MLP network.

To-do:
"""
import os
import argparse
import model_global
import er_mlp_max_margin
from config_parser import ConfigParser

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Build, run, and test ER MLP.')

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

    params = {
        'WORD_EMBEDDING': configparser.getbool('WORD_EMBEDDING'),
        'TRAINING_EPOCHS': configparser.getint('TRAINING_EPOCHS'),
        'BATCH_SIZE': configparser.getint('BATCH_SIZE'),
        'DISPLAY_STEP': configparser.getint('DISPLAY_STEP'),
        'EMBEDDING_SIZE': configparser.getint('EMBEDDING_SIZE'),
        'LAYER_SIZE': configparser.getint('LAYER_SIZE'),
        'LEARNING_RATE': configparser.getfloat('LEARNING_RATE'),
        'CORRUPT_SIZE': configparser.getint('CORRUPT_SIZE'),
        'LAMBDA': configparser.getfloat('LAMBDA'),
        'OPTIMIZER': configparser.getint('OPTIMIZER'),
        'ACT_FUNCTION': configparser.getint('ACT_FUNCTION'),
        'ADD_LAYERS': configparser.getint('ADD_LAYERS'),
        'DROP_OUT_PERCENT': configparser.getfloat('DROP_OUT_PERCENT'),
        'DATA_PATH': configparser.getstr('DATA_PATH'),
        'SAVE_MODEL': configparser.getbool('SAVE_MODEL'),
        'MODEL_SAVE_DIRECTORY': model_save_dir,
        'TRAIN_FILE': configparser.getstr('TRAIN_FILE'),
        'SEPARATOR': configparser.getstr('SEPARATOR'),
        'F1_FOR_THRESHOLD': configparser.getbool('F1_FOR_THRESHOLD'),
        'USE_SMOLT_SAMPLING': configparser.getbool('USE_SMOLT_SAMPLING'),
        'LOG_REG_CALIBRATE': configparser.getbool('LOG_REG_CALIBRATE'),
        'MARGIN': configparser.getfloat('MARGIN')
    }

    # run the model
    er_mlp_max_margin.run_model(params)

if __name__ == '__main__':
    main()
