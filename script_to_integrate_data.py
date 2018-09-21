#!/usr/bin/python

# import from generic packages
import sys
import numpy as np
import pandas as pd

# import from knowledge_scholar package
from modules.data_manager                              import integrate_data
from modules.inconsistency_manager                     import detect_inconsistencies
from modules.report_manager                            import plot_trustworthiness, save_resolved_inconsistencies, save_integrated_data
from modules.inconsistency_correctors.averagelog       import AverageLog
  
# COMMAND LINE ARGUMENTS
data_path_file          = sys.argv[1]
map_file                = sys.argv[2]
data_rule_file          = sys.argv[3]
inconsistency_rule_file = sys.argv[4]
data_out_file           = sys.argv[5]
inconsistency_out_file  = sys.argv[6]

# INTEGRATE DATA FROM MULTIPLE SOURCES
pd_data_paths           = pd.read_csv(data_path_file, sep = '\t', comment = '#')
pd_data                 = integrate_data(pd_data_paths, map_file, data_rule_file)

# DETECT INCONSISTENCIES
inconsistencies         = detect_inconsistencies(inconsistency_rule_file, pd_data)

# CORRECT INCONSISTENCIES
inconsistencies_with_max_belief, pd_belief_vector_without_inconsistencies, np_trustworthiness_vector = AverageLog.resolve_inconsistencies(pd_data, inconsistencies)

# REPORT DATA INTEGRATION RESULTS
plot_trustworthiness(pd_data, np_trustworthiness_vector, inconsistencies)

# SAVE INCONSISTENCIES
save_resolved_inconsistencies(inconsistency_out_file, inconsistencies_with_max_belief)

# SAVE INTEGRATED DATA
save_integrated_data(data_out_file, pd_belief_vector_without_inconsistencies)
