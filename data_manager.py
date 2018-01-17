#!/usr/bin/python

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def _convert_data(pd_data, source_name, map_file):
   pd_map = pd.read_csv(map_file, '\t')
   if 'Predicate' not in list(pd_data.columns.values):
      pd_data['Predicate'] = 'has'
   for idx, row in pd_data.iterrows():
      for column_name in list(pd_data.columns.values):
         mapped = pd_map[pd_map[source_name] == row[column_name]]['Destination']
         if np.any(mapped) and len(mapped) == 1:
            pd_data[idx][column_name] = mapped
   return pd_data

def _apply_data_rule(pd_data, data_rule_file):
   data_rules = ET.parse(data_rule_file).getroot()
   pd_new_data = pd_data.copy()
   for idx, row in pd_data.iterrows():
      for data_rule in data_rules:
         if_statement = data_rule.find('if')
         if condition_met(row, if_statement) == False:
            continue
         then_statement = data_rule.find('then')
         new_data = _change_data(row, then_statement)
         pd_new_data = pd_new_data.append(new_data)
   return pd_data

def _change_data(data, statement):
   new_data = data.copy()
   for feature in statement:
      feature_name = feature.get('name')
      feature_value = feature.get('value')
      new_data[feature_name] = feature_value
   return new_data
         
def integrate_data(data_path_file, map_file, data_rule_file):
   pd_data_paths = pd.read_csv(data_path_file, sep='\t', comment='#')
   list_integrated_data = []
   for idx, row in pd_data_paths.iterrows():
      pd_data = pd.read_csv(row['Path'],'\t')
      pd_data = _convert_data(pd_data, row['Source'], map_file)
      print("[data integration] adding source "+row['Source'])
      pd_data['Source'] = row['Source']
      list_integrated_data.append(pd_data)
   pd_integrated_data = pd.concat(list_integrated_data)
   return _apply_data_rule(pd_integrated_data, data_rule_file)

def condition_met(data, statement):
   if statement is None:
      return True
   condition_met = True
   for feature in statement:
      feature_name = feature.get('name')
      feature_value = feature.get('value')
      if data[feature_name] != feature_value:
         condition_met = False
   return condition_met
