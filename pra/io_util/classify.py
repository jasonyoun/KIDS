"""
Filename: classify.py

Authors:
	Nicholas Joodi - npjoodi@ucdavis.edu

Description:
	Using the best threshold, perform classification to find
	the predicted label.

To-do:
"""
import argparse

def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- parsed arguments
	"""
	parser = argparse.ArgumentParser(description='parse and generate the scores file')

	parser.add_argument(
		'--use_calibration',
		action='store_const',
		default=False,
		const=True)

	parser.add_argument(
		'--predicate',
		nargs='?',
		required=True,
		help='the predicate that we will get the scores for')

	parser.add_argument(
		'--dir',
		metavar='dir',
		nargs='?',
		default='./',
		help='base directory')

	return parser.parse_args()

if __name__ == "__main__":
	args = parse_argument()
	relation = args.predicate
	use_calibration = args.use_calibration

	scores_file = args.dir + '/scores/' + relation

	thresholds_file = 'dev/thresholds/' + relation
	if use_calibration:
		thresholds_file = 'dev/thresholds_calibration/' + relation

	classifications_file = args.dir + '/classifications/' + relation

	with open(scores_file, "r") as _file:
		scores = _file.readlines()

	with open(thresholds_file, "r") as _file:
		threshold = float(_file.readline().strip())

	scores = [x.strip().split('\t') for x in scores]
	classifications = []

	with open(classifications_file, "w") as _file:
		for score in scores:
			if float(score[0]) > threshold:
				_file.write('1\t' + str(score[0]) + '\n')
			else:
				_file.write('0\t' + str(score[0]) + '\n')
