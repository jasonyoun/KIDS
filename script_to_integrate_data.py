"""
Filename: script_to_integrate_data.py

Authors:
	Minseung Kim - msgkim@ucdavis.edu

Description:
	Integrate the data.

To-do:
	1. use argparse to parse command-line arguments automatically
	2. add logging instead of using print
"""

#!/usr/bin/python

# import from generic packages
import sys
import numpy as np
import pandas as pd

# import from knowledge_scholar package
from modules.data_manager import integrate_data
from modules.inconsistency_manager import detect_inconsistencies
from modules.report_manager import plot_trustworthiness, save_resolved_inconsistencies, save_integrated_data
from modules.inconsistency_correctors.averagelog import AverageLog

# command line arguments
data_path_file = sys.argv[1]
map_file = sys.argv[2]
data_rule_file = sys.argv[3]
inconsistency_rule_file = sys.argv[4]
data_out_file = sys.argv[5]
inconsistency_out_file = sys.argv[6]

# integrate data from multiple sources
pd_data_paths = pd.read_csv(data_path_file, sep = '\t', comment = '#')
pd_data = integrate_data(pd_data_paths, map_file, data_rule_file)

# detect inconsistencies
inconsistencies = detect_inconsistencies(inconsistency_rule_file, pd_data)

# correct inconsistencies
resolve_inconsistencies_result = AverageLog.resolve_inconsistencies(pd_data, inconsistencies)
inconsistencies_with_max_belief, pd_belief_vector_without_inconsistencies, np_trustworthiness_vector = resolve_inconsistencies_result

# report data integration results
plot_trustworthiness(pd_data, np_trustworthiness_vector, inconsistencies)

# save inconsistencies
save_resolved_inconsistencies(inconsistency_out_file, inconsistencies_with_max_belief)

# save integrated data
save_integrated_data(data_out_file, pd_belief_vector_without_inconsistencies)
