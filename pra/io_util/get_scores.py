"""
Filename: get_scores.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Given a prediction for a graph, get the scores for each triple.

To-do:
"""
import pickle
import argparse
import numpy as np

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

    parser.add_argument(
        '--log_reg_calibrate',
        metavar='log_reg_calibrate',
        nargs='?',
        default='false',
        help='use logistic regression for calibration, isotonic regression otherwise')

    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_argument()
    relation = args.predicate
    use_calibration = args.use_calibration

    if use_calibration:
        log_reg_calibrate = True if args.log_reg_calibrate == 'true' else False

        with open('calibrations/' + relation + '.pkl', 'rb') as pickle_file:
            clf = pickle.load(pickle_file)

    queries_tail = args.dir + '/queriesR_tail/' + relation
    fname = args.dir + '/predictions/' + relation
    scores_file = args.dir + '/scores/' + relation

    with open(fname) as _file:
        lines = _file.readlines()

    with open(queries_tail, "r") as l_file:
        queries_tail = l_file.readlines()

    entities_scores_dic = {}

    for line in lines:
        words = line.split('\t')
        subject = words[0]

        if subject not in entities_scores_dic:
            entities_scores_dic[subject] = {}

        del words[0]
        del words[0]

        for score_and_entity in words:
            score_and_entity = score_and_entity.split(',')
            entity = score_and_entity[1].replace('*', '').strip()
            score = float(score_and_entity[0].strip())
            entities_scores_dic[subject][entity] = score

    scores = []
    valid = []

    for line in queries_tail:
        subject = line.split('\t')[0].strip()
        tail = line.split('\t')[1].strip()

        if tail in entities_scores_dic[subject]:
            scores.append(entities_scores_dic[subject][tail])
            valid.append(1)
        else:
            scores.append(0.0)
            valid.append(0)

    if use_calibration:
        scores_array = np.array(scores)
        valid_array = np.array(valid)
        indices, = np.where(valid_array[:] > 0.)
        the_scores = scores_array[indices].reshape(-1, 1)

        if log_reg_calibrate:
            scores = clf.predict_proba(the_scores)[:, 1]
        else:
            scores = clf.transform(the_scores.ravel())

        scores_array[indices] = scores
    else:
        scores_array = np.array(scores)
        valid_array = np.array(valid)

    # write the scores to the file
    with open(scores_file, "w") as _file:
        for i in range(np.shape(scores_array)[0]):
            _file.write(str(scores_array[i]) + '\t' + str(valid_array[i]) + '\n')


if __name__ == "__main__":
    main()
