import operator
import numpy as np
import pandas as pd
import math
from collections import Counter
import xml.etree.ElementTree as ET
from .data_manager import condition_met

SPO_LIST = ['Subject','Predicate','Object']

def _has_conflict_values(pd_data, conflict_feature_name, conflict_feature_values):
   all_feature_values = pd.unique(pd_data[conflict_feature_name])
   has_conflict_values = False
   for conflict_feature_value in conflict_feature_values:
      if conflict_feature_value in all_feature_values:
         has_conflict_values = True
   return has_conflict_values

def detect_inconsistencies(inconsistency_rule_file, pd_data):
   inconsistency_rules = ET.parse(inconsistency_rule_file).getroot()

   inconsistencies = []
   for inconsistency_rule in inconsistency_rules:
      inconsistency_rule_name = inconsistency_rule.get('name')
      condition_statement = inconsistency_rule.find('condition')

      inconsistency_statement = inconsistency_rule.find('inconsistency')
      conflict_feature_name = inconsistency_statement.get('name')
      conflict_feature_values = [inconsistency_feature.get('value') for inconsistency_feature in inconsistency_statement]

      if _has_conflict_values(pd_data, conflict_feature_name, conflict_feature_values) == False:
         continue

      rest_feature_names = [feature_name for feature_name in SPO_LIST if feature_name != conflict_feature_name]
      unique_tuples = pd_data[rest_feature_names].drop_duplicates()
   
      for idx, row in unique_tuples.iterrows():
         print("[inconsistency rule] "+inconsistency_rule_name+" checking tuples "+str(idx)+"/"+str(len(unique_tuples))+"\r",end='')
         if condition_met(row, condition_statement) == False:
            continue
         filtered_indices = (pd_data[rest_feature_names[0]] == row[rest_feature_names[0]]) & \
                            (pd_data[rest_feature_names[1]] == row[rest_feature_names[1]])
         feature_values = pd_data[filtered_indices][conflict_feature_name]
         conflicts = [feature_value for feature_value in feature_values if feature_value in conflict_feature_values]
         if len(conflicts) < 2:
         	continue
         conflict_tuples = []
         for conflict in conflicts:
            new_data = row.copy()
            new_data[conflict_feature_name] = conflict
            conflict_tuples.append(tuple(new_data[SPO_LIST].values))
         inconsistencies.append(conflict_tuples)
      print("\n",end='')
   print("[inconsistency detection summary] found "+str(len(inconsistencies))+" inconsistencies.")
   return inconsistencies

def measure_accuracy(resolved_inconsistencies, answer):
   correctly_resolved_inconsistencies = 0.0
   for resolved_inconsistency in resolved_inconsistencies:
      if is_correct(resolved_inconsistency, answer):
         correctly_resolved_inconsistencies = correctly_resolved_inconsistencies + 1
   return float(correctly_resolved_inconsistencies) / float(len(resolved_inconsistencies))

def is_correct(resolved_inconsistency, answer):
   indices = (answer['Subject'] == resolved_inconsistency[0]) & \
             (answer['Predicate'] == resolved_inconsistency[1]) & \
             (answer['Object'] == resolved_inconsistency[2])
   return np.any(indices)
