import operator
import numpy as np
import pandas as pd
import math
from collections import Counter
import xml.etree.ElementTree as ET
from .data_manager import get_pd_of_statement

SPO_LIST = ['Subject','Predicate','Object']

def _data_has_conflict_values(all_feature_values, conflict_feature_values):
   unique_feature_values  = pd.unique(all_feature_values)
   num_of_conflict_values = 0

   for conflict_feature_value in conflict_feature_values:
      if conflict_feature_value in unique_feature_values:
         num_of_conflict_values = num_of_conflict_values + 1

   return num_of_conflict_values > 1

def detect_inconsistencies(inconsistency_rule_file, pd_data):
   inconsistency_rules = ET.parse(inconsistency_rule_file).getroot()
   inconsistencies     = []

   for inconsistency_rule in inconsistency_rules:
      inconsistency_rule_name = inconsistency_rule.get('name')
      condition_statement     = inconsistency_rule.find('condition')

      # CHECK IF CONDITION IS MET
      pd_condition_specific_data = pd_data.copy()
      if condition_statement is not None:
         pd_condition_statement     = get_pd_of_statement(condition_statement)
         indices_meeting_condition  = (pd_data[pd_condition_statement.index] == pd_condition_statement).all(1).values
         pd_condition_specific_data = pd_data[indices_meeting_condition].copy()

         if pd_condition_specific_data.shape[0] == 0:
            print("[inconsistency rule] {} is skipped because there are no data meeting condition.".format(inconsistency_rule_name))
            continue

      # CHECK IF DATA HAS CONFLICTS
      inconsistency_statement = inconsistency_rule.find('inconsistency')
      conflict_feature_name   = inconsistency_statement.get('name')
      conflict_feature_values = [inconsistency_feature.get('value') for inconsistency_feature in inconsistency_statement]

      if _data_has_conflict_values(pd_data[conflict_feature_name], conflict_feature_values) == False:
         print("[inconsistency rule] {} is skipped because there are no conflicts.".format(inconsistency_rule_name))
         continue

      rest_feature_names = [feature_name for feature_name in SPO_LIST if feature_name != conflict_feature_name]
      pd_grouped_data    = pd_data.groupby(rest_feature_names)[conflict_feature_name].apply(set)

      def has_conflict_values(x, conflict_feature_values):
         return x.intersection(conflict_feature_values)

      pd_grouped_data = pd_grouped_data.apply(has_conflict_values, args = (set(conflict_feature_values), ))
      pd_grouped_data = pd_grouped_data[pd_grouped_data.apply(len) > 1]

      for row_idx in range(pd_grouped_data.shape[0]):
         pd_conflict_data = pd.Series(pd_grouped_data.index[row_idx], index = rest_feature_names)

         conflict_tuples = []
         for conflict_value in pd_grouped_data[row_idx]:
            pd_conflict_data[conflict_feature_name] = conflict_value
            sources = pd.unique(pd_condition_specific_data[(pd_condition_specific_data[SPO_LIST] == pd_conflict_data).all(1)]['Source'])
            conflict_tuples.append((tuple(pd_conflict_data[SPO_LIST]), sources.tolist()))
         inconsistencies.append(conflict_tuples)

   print("[inconsistency detection summary] found {} inconsistencies.".format(len(inconsistencies)))

   return inconsistencies

def measure_accuracy(resolved_inconsistencies, answers):
   correctly_resolved_inconsistencies = 0.0

   for resolved_inconsistency in resolved_inconsistencies:
      ### SUPER UGLY : NEED TO UPDAE THE CORRECTORS ###
      resolved_tuple = resolved_inconsistency[0] if type(resolved_inconsistency[0]) == tuple and type(resolved_inconsistency[0][0]) == str else resolved_inconsistency[0][0]
 
      pd_resolved_inconsistency = pd.Series(resolved_tuple, index = SPO_LIST)
      answer = answers[(pd_resolved_inconsistency == answers).all(1)]
      if (pd_resolved_inconsistency == answers).all(1).any():
         print('[YES]\t'+','.join(pd_resolved_inconsistency)+' <-> '+','.join(answer))
         correctly_resolved_inconsistencies = correctly_resolved_inconsistencies + 1
      else:
         print('[NO]\t'+','.join(pd_resolved_inconsistency)+' <-> '+','.join(answer))
   print(correctly_resolved_inconsistencies)
   print(len(resolved_inconsistencies))
   return float(correctly_resolved_inconsistencies) / float(len(resolved_inconsistencies))



