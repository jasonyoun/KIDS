"""
Filename: analyze_model_predictions.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu
	Jason Youn - jyoun@ucdavis.edu

Description:
	Analyze model predictions.

To-do:
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import configparser
directory = os.path.dirname(__file__)

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Analyze model predictions.')

	parser.add_argument(
		'--dir',
		metavar='dir',
		nargs='+',
		default='./',
		help='base directory')

	parser.add_argument(
		'--results_dir',
		metavar='dir',
		nargs='?',
		action='store',
		required=True,
		help='directory to store the model')

	return parser.parse_args()

def load_data_array(_file, root):
	my_file = os.path.join(root, _file)
	df = pd.read_csv(my_file,sep='\t',encoding ='latin-1',header=None, 
					  names = ["subject", "predicate", "object", "label"])
	df = df.loc[(df['predicate'] == "confers#SPACE#resistance#SPACE#to#SPACE#antibiotic") \
		| (df['predicate'] == "upregulated#SPACE#by#SPACE#antibiotic")  \
		| (df['predicate'] == "targeted#SPACE#by")  ]
	df = df.loc[(df['label'] == 1 )]
	return df.as_matrix()

def load_confers_train_data_array(_file, root):
	my_file = os.path.join(root, _file)
	df = pd.read_csv(my_file,sep='\t',encoding ='latin-1',header=None, 
					  names = ["subject", "predicate", "object", "label"])
	df = df.loc[(df['predicate'] == "confers#SPACE#resistance#SPACE#to#SPACE#antibiotic") \
		| (df['predicate'] == "upregulated#SPACE#by#SPACE#antibiotic")  \
		| (df['predicate'] == "targeted#SPACE#by")  ]
	df = df.loc[(df['predicate'] == "confers#SPACE#resistance#SPACE#to#SPACE#antibiotic")  ]
	df = df.loc[df['label'] == 1 ]
	return df.as_matrix()

def get_edges_dic(array):
	edges_dic = {}
	for row in array:
		if row[2] not in edges_dic:
			edges_dic[row[2]]=0
		edges_dic[row[2]]+=1
	return edges_dic

def count_antibiotic_occurrence(antibiotic, edge_dic):
	count=0
	if antibiotic in edge_dic:
		count=edge_dic[antibiotic]
	return count

def create_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def load_df(_file, names):
	return pd.read_csv(_file,sep='\t',encoding ='latin-1',header=None, 
					  names = names)
def zero_if_nan(num):
	if  np.isnan(num):
		return 0
	else:
		return num

def calculate_recall(tp,fn):
	if tp+fn == 0:
		return 0
	else:
		return tp/(tp+fn)

def calculate_precision(tp,fp):
	if tp+fp == 0:
		return 0
	else:
		return tp/(tp+fp)

def calculate_f1(recall,precision):
	if precision+recall == 0:
		return 0
	else:
		return (2*precision*recall)/(precision+recall)


def get_stats(total_test, antibiotic):
	er_prediction = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_er_mlp == 1)].count()['object']
	stacked_prediction = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_stacked == 1)].count()['object']
	pra_prediction = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_pra == 1)].count()['object']

	er_fp= total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_er_mlp == 1)].count()['object']
	stacked_fp = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_stacked == 1)].count()['object']
	pra_fp = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_pra == 1)].count()['object']

	er_tn= total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_er_mlp == 0)].count()['object']
	stacked_tn = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_stacked == 0)].count()['object']
	pra_tn = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_pra == 0)].count()['object']

	er_fn= total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_er_mlp == 0)].count()['object']
	stacked_fn = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_stacked == 0)].count()['object']
	pra_fn = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_pra == 0)].count()['object']

	er_precision= calculate_precision(er_prediction,er_fp)
	stacked_precision = calculate_precision(stacked_prediction,stacked_fp)
	pra_precision = calculate_precision(pra_prediction,pra_fp)

	er_recall= calculate_recall(er_prediction,er_fn)
	stacked_recall = calculate_recall(stacked_prediction,stacked_fn)
	pra_recall = calculate_recall(pra_prediction,pra_fn)

	er_f1= calculate_f1(er_recall,er_precision)
	stacked_f1 = calculate_f1(stacked_recall,stacked_recall)
	pra_f1 = calculate_f1(pra_recall,pra_precision)
	return er_prediction,stacked_prediction,pra_prediction,er_fp,stacked_fp,pra_fp,\
		er_tn,stacked_tn,pra_tn,er_fn,stacked_fn,pra_fn,er_precision,stacked_precision, \
		pra_precision,er_recall,stacked_recall,pra_recall,er_f1,stacked_f1,pra_f1


if __name__ == '__main__':
	# parse args
	args = parse_argument()
	list_directories = args.dir
	results_dir = args.results_dir

	# config parser
	config = configparser.ConfigParser()

	# variables
	root_dir = os.path.abspath(os.path.join(directory, '../..'))

	create_dir(results_dir)

	_file =  os.path.join(results_dir, "edges_count.txt")
	stats_file =  os.path.join(results_dir, "test_stats.txt")
	with open(_file, 'w') as f_ec, open(stats_file, 'w') as f_ts:
		f_ec.write('Antibiotic'+'\t'+'genes_resist'+'\t'+'test_edges'+'\t'+'dev_edges'+'\t'+'edges_in_train'+'\t'+'confer_edges_in_train'+'\n')
		f_ts.write('Antibiotic'+'\t'+'genes_resist'+'\t'+'test_edges'+'\t'+'dev_edges'+'\t'+'edges_in_train'+'\t'+'confer_edges_in_train'+'\t'+'er_tp'+'\t'+'er_tn'+'\t'+'er_fp'+'\t'+'pra_tp'+'\t'+'pra_tn'+'\t'+'pra_fp'+'\t'+'stacked_tp'+'\t'+'stacked_tn'+'\t'+'stacked_fp'+'\t'+'er_precision'+'\t'+'pra_precision'+'\t'+'stacked_precision'+'\t'+'er_recall'+'\t'+'pra_recall'+'\t'+'stacked_recall'+'\t'+'er_f1'+'\t'+'pra_f1'+'\t'+'stacked_f1'+'\n')

		for fold in list_directories:
			print('Processing fold {}...'.format(fold))
			pra_classifications = root_dir+'/pra/model/model_instance/'+fold+'/instance/test/classifications_pra.txt'
			er_mlp_classifications = root_dir+'/er_mlp/model/model_instance/'+fold+'/test/classifications_er_mlp.txt'
			stacked_classifications = root_dir+'/stacked/model_instance/'+fold+'/test/classifications_stacked.txt'
			configuration=root_dir+'/run/configuration/'+fold+'/er_mlp.ini'
			config.read(configuration)
			data_path=config['DEFAULT']['DATA_PATH']

			train_confers_array = load_confers_train_data_array('train.txt',data_path)
			test_array = load_data_array('test.txt',data_path)
			dev_array = load_data_array('dev.txt',data_path)
			data_array = load_data_array('data.txt',data_path+'/../')
			train_array = load_data_array('train.txt',data_path)

			data_antibiotic_confers_dic = get_edges_dic(data_array)
			test_edges_dic = get_edges_dic(test_array)
			dev_edges_dic = get_edges_dic(dev_array)
			train_confers_dic = get_edges_dic(train_confers_array)
			train_edges_dic = get_edges_dic(train_array)

			for k,v in data_antibiotic_confers_dic.items():
				test_count=count_antibiotic_occurrence(k,test_edges_dic)
				dev_count=count_antibiotic_occurrence(k,dev_edges_dic)
				train_count=count_antibiotic_occurrence(k,train_edges_dic)
				train_confers_count=count_antibiotic_occurrence(k,train_confers_dic)
				f_ec.write(k+'\t'+str(v)+'\t'+str(test_count)+'\t'+str(dev_count)+'\t'+str(train_count)+'\t'+str(train_confers_count)+'\n')

			_file = "test.txt"
			test_file = os.path.join(data_path, _file)
			test_df = load_df(_file=test_file,names = ["subject", "predicate", "object", "label"])
			er_df = load_df(_file=er_mlp_classifications,names = ["classification_er_mlp"])
			st_df = load_df(_file=stacked_classifications,names = ["classification_stacked"])
			pra_df = load_df(_file=pra_classifications,names = ["classification_pra",'score'])
			pra_df =pra_df.classification_pra
			total_test = pd.concat([test_df, er_df,st_df,pra_df],axis=1)
			test_antibiotic = test_df.object.tolist()
			test_df=test_df.loc[(test_df['label'] == 1)  ]
			test_antibiotic= set(test_df.object.tolist())

			for antibiotic in test_antibiotic:
				er_prediction,stacked_prediction,pra_prediction,er_fp,stacked_fp,pra_fp,er_tn,stacked_tn,pra_tn,er_fn,stacked_fn,pra_fn,er_precision,stacked_precision, pra_precision,er_recall,stacked_recall,pra_recall,er_f1,stacked_f1,pra_f1=get_stats(total_test,antibiotic)
				genes_resist_count=count_antibiotic_occurrence(antibiotic,data_antibiotic_confers_dic)
				test_count=count_antibiotic_occurrence(antibiotic,test_edges_dic)
				dev_count=count_antibiotic_occurrence(antibiotic,dev_edges_dic)
				train_count=count_antibiotic_occurrence(antibiotic,train_edges_dic)
				train_confers_count=count_antibiotic_occurrence(antibiotic,train_confers_dic)
				f_ts.write(antibiotic+'\t'+str(genes_resist_count)+'\t'+str(test_count)+'\t'+str(dev_count)+'\t'+str(train_count)+'\t'+str(train_confers_count)+'\t'+str(er_prediction)+'\t'+str(er_tn)+'\t'+str(er_fp)+'\t'+str(pra_prediction)+'\t'+str(pra_tn)+'\t'+str(pra_fp)+'\t'+str(stacked_prediction)+'\t'+str(stacked_tn)+'\t'+str(stacked_fp)+'\t'+str(er_precision)+'\t'+str(pra_precision)+'\t'+str(stacked_precision)+'\t'+str(er_recall)+'\t'+str(pra_recall)+'\t'+str(stacked_recall)+'\t'+str(er_f1)+'\t'+str(pra_f1)+'\t'+str(stacked_f1)+'\n')
