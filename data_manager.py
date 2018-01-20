#!/usr/bin/python

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

SPO_LIST = ['Subject', 'Predicate', 'Object']

def integrate_data(data_path_file, map_file, data_rule_file):
   pd_data_paths        = pd.read_csv(data_path_file, sep='\t', comment='#')
   list_integrated_data = []

   for idx, row in pd_data_paths.iterrows():
      pd_data           = pd.read_csv(row['Path'],'\t')
      pd_data           = _convert_data(pd_data, row['Source'], map_file)
      pd_data['Source'] = row['Source']

      list_integrated_data.append(pd_data)
      print("[data integration] adding {} tuples from source {}".format(pd_data.shape[0],row['Source']))

   pd_integrated_data = pd.concat(list_integrated_data)
   pd_integrated_data = _apply_data_rule(pd_integrated_data, data_rule_file)
   print("[data integration] finally {} tuples are integrated.".format(pd_integrated_data.shape[0]))
   
   return pd_integrated_data

def _convert_data(pd_data, source_name, map_file):
   pd_map = pd.read_csv(map_file, '\t')

   if 'Predicate' not in list(pd_data.columns.values):
      pd_data['Predicate'] = 'has'
   
   for idx, row in pd_map.iterrows():
      indices          = (pd_data[SPO_LIST] == row[source_name]).values
      pd_data[indices] = row['Destination']

   return pd_data


def _apply_data_rule(pd_data, data_rule_file):
   data_rules   = ET.parse(data_rule_file).getroot()
   pd_new_data  = pd_data.copy()

   for data_rule in data_rules:
      if_statement              = data_rule.find('if')
      pd_if_statement           = get_pd_of_statement(if_statement)
      indices_meeting_if        = (pd_data[pd_if_statement.index] == pd_if_statement).all(1).values
      pd_rule_specific_new_data = pd_data[indices_meeting_if].copy()

      if pd_rule_specific_new_data.shape[0] == 0:
         continue

      then_statement    = data_rule.find('then')
      pd_then_statement = get_pd_of_statement(then_statement)
      pd_rule_specific_new_data[pd_then_statement.index] = pd_then_statement.tolist()
      pd_new_data       = pd_new_data.append(pd_rule_specific_new_data)
   
   print("[data integration] {} new tuples are added based on data rule.".format(pd_new_data.shape[0]))
   
   return pd_new_data

def get_pd_of_statement(statement):
   feature_names   = [feature.get('name') for feature in statement]
   feature_values  = [feature.get('value') for feature in statement]
   
   return pd.Series(feature_values, index=feature_names)
