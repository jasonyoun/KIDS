"""
Filename: visualize.py

Authors:
	Jason Youn -jyoun@ucdavis.edu

Description:
	Visualize data in multiples ways as requested by the user.

To-do:
"""

#!/usr/bin/python

# import from generic packages
import os
import sys
import warnings
import argparse
import openpyxl
import numpy as np
import pandas as pd
import logging as log
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as plticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import SpectralBiclustering


# default arguments
DEFAULT_DATA_DIR_STR = '../output'
DEFAULT_ENTITY_FULL_NAMES_FILENAME = 'entity_full_names.txt'
DEFAULT_KNOWLEDGE_GRAPH_FILENAME = 'out.txt'

# drop these columns because it's not necessary for network training
COLUMN_NAMES_TO_DROP = ['Belief', 'Source size', 'Sources']


def set_logging():
	"""
	Configure logging.
	"""
	log.basicConfig(format='(%(levelname)s) %(filename)s: %(message)s', level=log.DEBUG)

	# set logging level to WARNING for matplotlib
	logger = log.getLogger('matplotlib')
	logger.setLevel(log.WARNING)

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Visualize data.')

	parser.add_argument(
		'--data_dir',
		default=DEFAULT_DATA_DIR_STR,
		help='Path to the file data_path_file.txt')

	return parser.parse_args()

def save_cra_matrix(pd_known_cra, pd_genes, pd_antibiotics):
	pd_pivot_table = pd.pivot_table(
						pd_known_cra[['Subject', 'Predicate', 'Object']],
						index='Subject',
						columns='Object',
						values='Predicate',
						aggfunc='first')

	genes_not_in_cra_triples = list(set(pd_genes.values) - set(pd_pivot_table.index.values))
	antibiotics_not_in_cra_triples = list(set(pd_antibiotics.values) - set(pd_pivot_table.columns.values))

	if len(genes_not_in_cra_triples) > 0:
		pd_pivot_table = pd_pivot_table.reindex(index=pd_pivot_table.index.union(genes_not_in_cra_triples))

	if len(antibiotics_not_in_cra_triples) > 0:
		pd_pivot_table = pd_pivot_table.reindex(columns=pd_pivot_table.columns.union(antibiotics_not_in_cra_triples))

	# pd_pivot_table.to_excel('/home/jyoun/Jason/UbuntuShare/heatmap.xlsx')

	return pd_pivot_table

def plot_heatmap(pd_pivot_table):
	pd_pivot_table = pd_pivot_table.replace('confers resistance to antibiotic', 1)
	pd_pivot_table = pd_pivot_table.replace('confers no resistance to antibiotic', -1)
	pd_pivot_table = pd_pivot_table[pd_pivot_table.columns].astype(float)
	pd_pivot_table = pd_pivot_table.fillna(0)

	# compute the required figure height
	top_margin = 0.02
	bottom_margin = 0.12

	# build the figure instance with the desired height
	fig, ax = plt.subplots(
			figsize=(50, 60),
			gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin))

	# do spectral biclustering on the data
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		model = SpectralBiclustering(n_clusters=(2, 2))
		model.fit(pd_pivot_table.values)

	# restore indexes and columns of biclustering result
	row_idx_0, col_idx_0 = model.get_indices(0)
	_, col_idx_1 = model.get_indices(1)
	row_idx_2, _ = model.get_indices(2)

	new_idx = [pd_pivot_table.index[i] for i in list(np.concatenate((row_idx_0, row_idx_2)))]
	new_columns = [pd_pivot_table.columns[i] for i in list(np.concatenate((col_idx_0, col_idx_1)))]
	pd_pivot_table = pd_pivot_table.reindex(index=new_idx, columns=new_columns)

	# generate heatmap
	myColors = [(1, 0.6, 0.6), (0.93, 0.93, 0.93), (0, 0, 0.7)]
	cmap = LinearSegmentedColormap.from_list('custom', myColors, len(myColors))
	ax = sns.heatmap(pd_pivot_table, cmap=cmap, ax=ax, cbar_kws={"shrink": 0.75})
	# ax.collections[0].colorbar.remove()

	# colorbar
	cbar = ax.collections[0].colorbar
	cbar.ax.get_yaxis().set_ticks([])

	for i, lab in enumerate(['confers no resistance to antibiotic', 'unknown', 'confers resistance to antibiotic']):
	    cbar.ax.text(0, (i-1) * 0.667, lab, ha='center', va='center', fontsize=50, rotation=270, color='white' if i != 1 else 'black')

	# x, y axis labels
	ax.set_xlabel('Antibiotics', size=70)
	ax.set_ylabel('Genes', size=70)
	ax.xaxis.set_label_coords(0.5, -0.04)
	ax.yaxis.set_label_coords(-0.02, 0.5)
	ax.set_yticks([])

	plt.savefig('/home/jyoun/Jason/UbuntuShare/fig1.pdf')


if __name__ == '__main__':
	# set log and parse args
	set_logging()
	args = parse_argument()

	# read full names dataframe
	pd_entities = pd.read_csv(
		os.path.join(args.data_dir, DEFAULT_ENTITY_FULL_NAMES_FILENAME),
		sep='\n',
		names=['Full name'])

	# split full names
	pd_entities = pd_entities['Full name'].str.split(':', n=2, expand=True)
	pd_entities.columns = ['Global', 'Type', 'Name']

	# get all genes and antibiotics in KG
	pd_genes = pd_entities[pd_entities['Type'].str.match('gene')]['Name'].reset_index(drop=True)
	pd_antibiotics = pd_entities[pd_entities['Type'].str.match('antibiotic')]['Name'].reset_index(drop=True)

	pd_genes.sort_values(inplace=True)
	pd_antibiotics.sort_values(inplace=True)

	log.debug('Number of genes: {}'.format(pd_genes.shape[0]))
	log.debug('Number of antibiotics: {}'.format(pd_antibiotics.shape[0]))

	# read knowledge graph and drop unnecessary columns
	pd_kg = pd.read_csv(os.path.join(args.data_dir, DEFAULT_KNOWLEDGE_GRAPH_FILENAME), sep='\t')
	pd_kg = pd_kg.drop(COLUMN_NAMES_TO_DROP, axis=1)

	pd_known_cra = pd_kg[pd_kg['Predicate'].str.contains('resistance to antibiotic')].reset_index(drop=True)
	if pd_known_cra.duplicated(keep='first').sum() != 0:
		log.warning('There are duplicates in the knowledge graph!')

	log.debug('Number of CRA triples: {}'.format(pd_known_cra.shape[0]))



	# save heatmap as matrix
	pd_pivot_table = save_cra_matrix(pd_known_cra, pd_genes, pd_antibiotics)

	plot_heatmap(pd_pivot_table)

