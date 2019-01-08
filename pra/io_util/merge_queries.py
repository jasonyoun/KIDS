"""
Filename: merge_queries.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu

Description:
	Merge queries.

To-do:
"""
import pandas as pd
import os.path
import argparse

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='evaluate the results')

	parser.add_argument(
		'--dir',
		metavar='dir',
		nargs='?',
		default='./',
		help='base directory')

	return parser.parse_args()

if __name__ == "__main__":
	args = parse_argument()

	with open('./selected_relations') as f:
		relations = f.readlines()

	relations = [x.strip() for x in relations]
	_relations = []

	for r in relations:
		_relations.append('_' + r)

	relations += _relations

	for r in relations:
		pos_df = pd.read_csv(args.dir + '/queriesR_train/' + r, sep='\t', encoding='latin-1', names=["subject", "object"])
		if not os.path.isfile(args.dir + '/queriesR_train_neg/' + r):
			continue

		neg_df = pd.read_csv(args.dir + '/queriesR_train_neg/' + r, sep='\t', encoding='latin-1', names=["subject", "object"])
		result = pd.merge(pos_df, neg_df, how='left', on="subject")
		result.to_csv(args.dir + '/queriesR_train/' + r, sep='\t', index=False, header=False)
