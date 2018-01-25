#!/usr/bin/python

# import generic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, rcParams

COLUMN_NAMES = ['Subject','Predicate','Object','Source']

def plot_data_summary(pd_data, inconsistencies, np_trustworthiness_vector):
   pd_data_copy = pd_data.copy()
   pd_data_copy['Inconsistency'] = 'No'
   
   for inconsistency_tuples in inconsistencies:
      for inconsistency_tuple, sources in inconsistency_tuples:
         for source in sources:
            pd_inconsistency_tuple = pd.Series(inconsistency_tuple + (source,),index=COLUMN_NAMES)
            found_tuple = pd_data[(pd_data[COLUMN_NAMES] == pd_inconsistency_tuple).all(1)]
            pd_data_copy.at[found_tuple.index.values[0],'Inconsistency'] = 'Yes'

   pd_grouped_data = pd_data_copy[pd_data_copy['Inconsistency'] == 'No'].groupby(['Subject','Predicate','Object'])['Source'].apply(set)
   sources = pd.unique(pd_data_copy['Source']).tolist()

   pd_data_stat_column_names = ['Inconsistency', 'Single source', 'Multiple sources']
   pd_data_stat = pd.DataFrame(index=sources, columns=pd_data_stat_column_names)
   
   for source in sources:
     matched_indices = (pd_data_copy[['Source','Inconsistency']] == pd.Series([source, 'Yes'], index=['Source','Inconsistency'])).all(1)
     num_of_inconsistencies = sum(matched_indices)
     num_of_tuples_with_one_source = sum(pd_grouped_data == {source})
     num_of_rest = sum(pd_data_copy['Source'] == source) - num_of_inconsistencies - num_of_tuples_with_one_source

     pd_data_stat.loc[source] = [num_of_inconsistencies, num_of_tuples_with_one_source, num_of_rest]
    
   generate_sankey_data(pd_data_copy)

   rcParams['font.family'] = 'sans-serif'
   rcParams['font.sans-serif'] = ['Sans']
   fig, ax1 = plt.subplots()
   ax1.set_yscale("log")

   dim  = len(pd_data_stat_column_names)
   w    = 0.75
   dimw = w / dim
   
   x = np.arange(len(sources))
   i = 0
   for column_name in pd_data_stat_column_names:
      ax1.bar(x + i * dimw, pd_data_stat[column_name], dimw, label = column_name, bottom = 0.001)
      i = i + 1

   ax2 = ax1.twinx()
   list_trustworthiness_vector = [float(trustworthiness) for trustworthiness in np_trustworthiness_vector]
   ax2.plot(x + dimw, list_trustworthiness_vector, marker = 'o', color = 'r', label = 'Trustworthiness')
   ax1.plot(np.nan, marker = 'o', color = 'r', label = 'Trustworthiness')

   ax1.set_xlabel('Sources')
   ax1.set_ylabel('Number of tuples')
   ax1.set_xticks(x + dimw)
   ax1.set_xticklabels(sources)
   
   ax2.set_ylabel('Trustworthiness')
   ax2.set_xticks(x + dimw)
   ax1.legend()
   plt.savefig('data_summary.png')
   plt.close()

def generate_sankey_data(pd_data):
   pd_grouped_data = pd_data.groupby(['Source', 'Predicate', 'Inconsistency']).size()
   out_sankey_data = open('sankey_data_summary.txt','w')

   out_sankey_data.write(','.join(['source','type','target','value'])+'\n')
   for index in pd_grouped_data.index:
      out_sankey_data.write("{},{},{},{}\n".format(index[0],index[1],index[2],pd_grouped_data.loc[index]))