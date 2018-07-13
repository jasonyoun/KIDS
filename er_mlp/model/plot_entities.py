import numpy as np
import pickle as pickle
import pandas as pd
import sys
import os
directory = os.path.dirname(__file__)
print(__file__)
print(directory)
import configparser
abs_path_data= os.path.join(directory, '../data_handler')
sys.path.insert(0, abs_path_data)
abs_path_er_mlp= os.path.join(directory, '../er_mlp_imp')
sys.path.insert(0, abs_path_er_mlp)
import tensorflow as tf
from sklearn import utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import random
from tensorflow.python import debug as tf_debug
from scipy import interp
import random
from er_mlp import ERMLP
abs_path_metrics= os.path.join(directory, '../../utils')
sys.path.insert(0, abs_path_metrics)
from data_processor import DataProcessor
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
if directory != '':
    directory = directory+'/'
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from data_orchestrator_cm import DataOrchestrator
import  data_orchestrator_cm 
import matplotlib.patches as mpatches
import pylab as plot
import argparse
from sklearn.decomposition import PCA

font = {
        'size'   : 15}

matplotlib.rc('font', **font)

config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description='evaluate')
parser.add_argument('--dir', metavar='dir', nargs='?', default='./',
                    help='base directory')
parser.add_argument('--use_calibration',action='store_const',default=False,const=True)



args = parser.parse_args()
calibrated = args.use_calibration
model_instance_dir='model_instance/'
model_save_dir=model_instance_dir+args.dir
configuration = model_save_dir+'/'+args.dir.replace('/','')+'.ini'

config.read('./'+configuration)
WORD_EMBEDDING = config.getboolean('DEFAULT','WORD_EMBEDDING')
MODEL_SAVE_DIRECTORY=model_save_dir
DATA_PATH=config['DEFAULT']['DATA_PATH']
WORD_EMBEDDING = config.getboolean('DEFAULT','WORD_EMBEDDING')
DATA_TYPE = config['DEFAULT']['DATA_TYPE']
TRAINING_EPOCHS = config.getint('DEFAULT','TRAINING_EPOCHS')
BATCH_SIZE = config.getint('DEFAULT','BATCH_SIZE')
DISPLAY_STEP =  config.getint('DEFAULT','DISPLAY_STEP')
EMBEDDING_SIZE = config.getint('DEFAULT','EMBEDDING_SIZE')
LAYER_SIZE = config.getint('DEFAULT','LAYER_SIZE')
LEARNING_RATE = config.getfloat('DEFAULT','LEARNING_RATE')
CORRUPT_SIZE = config.getint('DEFAULT','CORRUPT_SIZE')
LAMBDA = config.getfloat('DEFAULT','LAMBDA')
OPTIMIZER = config.getint('DEFAULT','OPTIMIZER')
ACT_FUNCTION = config.getint('DEFAULT','ACT_FUNCTION')
ADD_LAYERS = config.getint('DEFAULT','ADD_LAYERS')
DROP_OUT_PERCENT = config.getfloat('DEFAULT','ADD_LAYERS')





def plot_with_labels(low_dim_embs, colors,labels, filename='pca_entities_random.png'):
    fig =plt.figure(figsize=(18, 18)) 
    ax = fig.add_subplot(111, projection='3d')
    label = ['gene','molecular function','biological process','cellular component','antibiotic']
    col = ['r','b','g','m','y']
    recs = []
    for i in range(0,len(col)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=col[i]))
    for i, color in enumerate(colors):
        x, y,z = low_dim_embs[i, :]
        if color=='r':
            index =0
        if color=='b':
            index =1
        if color=='g':
            index =2
        if color=='m':
            index =3
        if color=='y':
            index =4
        ax.scatter(x, y,z,c=color,label=label[index])
    ax.legend(recs,label,loc="lower right")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.title("Entity Embedding PCA Dimensional Reduction ")
    plt.savefig(filename)
def create_predicate_to_entity_to_triplets_set(data_array ):
    predicate_to_entity_to_triplets = {}
    for row in data_array:
        if row[3] == 1:
            if row[1] not in predicate_to_entity_to_triplets:
                predicate_to_entity_to_triplets[row[1]] = {}
            if row[2] not in predicate_to_entity_to_triplets[row[1]]:
                predicate_to_entity_to_triplets[row[1]][row[2]] = set()
            predicate_to_entity_to_triplets[row[1]][row[2]].add(row[0])
    return predicate_to_entity_to_triplets

print("begin tensor seesion")
with tf.Session() as sess:

    processor = DataProcessor()
    saver = tf.train.import_meta_graph(MODEL_SAVE_DIRECTORY+'/model.meta')
    saver.restore(sess, MODEL_SAVE_DIRECTORY+'/model')
    fn = open(MODEL_SAVE_DIRECTORY+'/params.pkl','rb')
    params = pickle.load(fn)
    entity_dic = params['entity_dic']
    pred_dic = params['pred_dic']
    thresholds = params['thresholds']
    indexed_entities = params['indexed_entities']
    if calibrated:
        calibration_models = params['calibrated_models']
        thresholds = params['thresholds_calibrated']
        print(calibration_models)
    num_preds = len(pred_dic)
    num_entities= len(entity_dic)
    graph = tf.get_default_graph()
    y = graph.get_tensor_by_name("y:0")
    triplets = graph.get_tensor_by_name("triplets:0")
    E = graph.get_tensor_by_name("E:0")
    P = graph.get_tensor_by_name("P:0")
    predictions = tf.get_collection('predictions')[0]
    E = E.eval()

    # # hello = [np.mean(E[ ent_word_indice,:],0) for ent_word_indice in indexed_entities]
    # print(len(hello))
    entity_embeddings= np.vstack([np.mean(E[ np.array(ent_word_indice),:],0) for ent_word_indice in indexed_entities])
    # E=entity_emb
    print('entity_embeddings')
    print(np.shape(entity_embeddings))
    print('len(entity_dic)')
    print(len(entity_dic))


    er_mlp_params = {
        'word_embedding': WORD_EMBEDDING,
        'embedding_size': EMBEDDING_SIZE,
        'layer_size': LAYER_SIZE,
        'corrupt_size': CORRUPT_SIZE,
        'lambda': LAMBDA,
        'num_entities':num_entities,
        'num_preds':num_preds,
        'learning_rate':LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'add_layers': ADD_LAYERS,
        'act_function':ACT_FUNCTION,
        'drop_out_percent': DROP_OUT_PERCENT
    }

    if WORD_EMBEDDING:
        num_entity_words = params['num_entity_words']
        num_pred_words = params['num_pred_words']
        indexed_entities = params['indexed_entities']
        indexed_predicates = params['indexed_predicates']
        er_mlp_params['num_entity_words'] = num_entity_words
        er_mlp_params['num_pred_words'] = num_pred_words
        er_mlp_params['indexed_entities'] = indexed_entities
        er_mlp_params['indexed_predicates'] = indexed_predicates

    df = processor.load(DATA_PATH+'train.txt')
    indexed_train_data = processor.create_indexed_triplets_test(df.as_matrix(),entity_dic,pred_dic )
    data_orch = DataOrchestrator( indexed_train_data, DATA_PATH,pred_dic,entity_dic, corruption_size=10)
    type_dic,subsets_dic = data_orch.get_type_subsets_dic()
    er_mlp = ERMLP(er_mlp_params)
    predicate_to_entity_to_triplets_set = create_predicate_to_entity_to_triplets_set(df.as_matrix() )
    antibiotics_to_genes = predicate_to_entity_to_triplets_set['confers#SPACE#resistance#SPACE#to#SPACE#antibiotic']
    id_entity_dic = {}
    for k,v in entity_dic.items():
        id_entity_dic[v] = k

    entity_embeddings_keep =[]
    embedding = np.array([])
    entitys = [None] * len(entity_dic)
    labels = [None] * len(entity_dic)
    count=0
    for k in subsets_dic['gene']:
        entitys.append('r')
        labels.append('gene')
        entity_embeddings_keep.append(entity_embeddings[k])
    count=0
    for k in subsets_dic['molecular_function']:
        entitys.append('b')
        labels.append('molecular_function')
        entity_embeddings_keep.append(entity_embeddings[k])

    count=0
    for k in subsets_dic['biological_process']:
        entitys.append('g')
        labels.append('biological_process')
        entity_embeddings_keep.append(entity_embeddings[k])

    count=0
    for k in subsets_dic['cellular_component']:
        entitys.append('m')
        labels.append('cellular_component')
        entity_embeddings_keep.append(entity_embeddings[k])

    count=0
    for k in subsets_dic['antibiotic']:
        entitys.append('y')
        labels.append('antibiotic')
        entity_embeddings_keep.append(entity_embeddings[k])
    limit = len(entity_embeddings_keep)
    entitys = [x for x in entitys if x is not None]
    entity_embeddings = np.vstack(entity_embeddings_keep)

    vector_dim = 50
    # Reshaping embedding
    embedding = entity_embeddings.reshape(limit, vector_dim)
    pca = PCA(n_components=3)
    tsne = TSNE(perplexity=30.0, n_components=3, init='pca', n_iter=5000)
    low_dim_embedding = pca.fit_transform(embedding)
    # low_dim_embedding = tsne.fit_transform(embedding)
    plot_with_labels(low_dim_embedding, entitys,labels, filename=MODEL_SAVE_DIRECTORY+'/tsne_entities_classification.png')

