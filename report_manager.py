#!/usr/bin/python

# import generic packages
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import pyplot, rcParams
from scipy.stats import gamma

# import from knowledge_scholar
from .inconsistency_manager import get_belief_of_inconsistencies

# set fonts
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# constants
SPO_LIST     = ['Subject','Predicate','Object']
COLUMN_NAMES = SPO_LIST + ['Source']

# IT IS NOT GENERIC (NEED TO UPDATE)
data_category = { 'Soo': '', 'Liu': 'MIC profile', 'Shaw': 'Expression profile', 'Tamae': 'MIC profile', 'CARD': 'KB', 'GO': 'KB' }

def plot_network_of_inconsistency(inconsistencies):
   return 
   pd_nodes = pd.DataFrame(columns = ['name'])
   pd_edges = pd.DataFrame(columns = ['source','target'])

   pd_nodes_idx = 0
   pd_edges_idx = 0
   for inconsistent_tuples in inconsistencies.values():
      inconsistent_sources = []
      for inconsistent_tuple, sources in inconsistent_tuples:
         for source in sources:
            pd_nodes.loc[pd_nodes_idx] = source
            pd_nodes_idx = pd_nodes_idx + 1
            inconsistent_sources.append(source)
      s = inconsistent_sources[0]
      for t in inconsistent_sources[1:]:
         pd_edges.loc[pd_edges_idx] = [s, t]
         pd_edges_idx = pd_edges_idx + 1

   print(pd_nodes.head())
   print(pd_edges.head())
   print(pd_nodes.shape)
   print(pd_edges.shape)
   pd_grouped_nodes = pd.DataFrame(pd_nodes.groupby('name').size())
   for source in pd_grouped_nodes.index:
      pd_grouped_nodes.loc[source]['group'] = data_category[source]

   pd_grouped_edges = pd.DataFrame(pd_edges.groupby(['source', 'target']).size())

   print(pd_grouped_nodes)
   print(pd_grouped_edges)

def plot_pie_summary(pd_data, inconsistencies):
   inconsistency_source_category = pd.Series(index = inconsistencies.keys())
   '''
   for inconsistency_tuples in inconsistencies.values():
      for inconsistency_tuple, sources in inconsistency_tuples:
         for source in sources:
            inconsistency_source_category data_category[source]

   plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
 
   plt.axis('equal')  
   plt.savefig('integrated_data_summary.pdf')
   plt.close()    
   '''

def plot_trustworthiness(pd_data, np_trustworthiness_vector):
   pd_data_copy = pd_data.copy()
   
   pd_grouped_data = pd_data_copy.groupby(SPO_LIST)['Source'].apply(set)
   sources = pd_data.groupby('Source').size().index.tolist()
   list_trustworthiness_vector = [float(trustworthiness) for trustworthiness in np_trustworthiness_vector]
   pd_trustworthiness_vector = pd.Series(list_trustworthiness_vector, index = sources)

   pd_data_stat_column_names = ['Single source', 'Multiple sources']
   pd_data_stat = pd.DataFrame(index=sources, columns=pd_data_stat_column_names)
   
   for source in sources:
     matched_indices = (pd_data_copy['Source'] == source)
     num_of_tuples_with_one_source = sum(pd_grouped_data == {source})
     num_of_rest = sum(pd_data_copy['Source'] == source) - num_of_tuples_with_one_source

     pd_data_stat.loc[source] = [num_of_tuples_with_one_source, num_of_rest]
    
   fig, ax1 = plt.subplots()
   ax1.set_yscale("log")

   dim  = len(pd_data_stat_column_names)
   w    = 0.5
   dimw = w / dim
   
   sorted_sources = pd_trustworthiness_vector.sort_values(ascending=False).index.tolist()
   pd_data_stat = pd_data_stat.loc[sorted_sources]

   x = np.arange(len(sources))
   i = 0
   for column_name in pd_data_stat_column_names:
      ax1.bar(x + i * dimw + 0.5 * dimw, pd_data_stat[column_name], dimw, label = column_name, bottom = 0.001)
      i = i + 1

   ax2 = ax1.twinx()
   ax2.plot(x + dimw, pd_trustworthiness_vector[sorted_sources].tolist(), marker = 'o', color = 'r', label = 'Trustworthiness')
   ax1.plot(np.nan, marker = 'o', color = 'r', label = 'Trustworthiness')

   ax1.set_xlabel('Sources')
   ax1.set_ylabel('Number of tuples')
   ax1.set_xticks(x + dimw)
   ax1.set_xticklabels(sorted_sources)
   
   ax2.set_ylabel('Relative trustworthiness')
   ax2.set_xticks(x + dimw)
   ax1.legend()
   plt.savefig('trustworthiness_data_summary.pdf')
   plt.close()

def plot_belief_of_inconsistencies(inconsistencies_with_max_belief, answer, inconsistency_out_file_prefix):
   data   = get_belief_of_inconsistencies(inconsistencies_with_max_belief, answer)
   colors = ['b','g']
   labels = ['Correctly resolved tuple','Incorrectly resolved tuple']

   fig, ax = plt.subplots()
   ax.set_xlim(0,1)
   for i in data:
      data_points = data[i]

      x = np.linspace(-0.1, 1, 100)

      ax.hist(data_points, normed=True, label=labels[i], histtype='stepfilled', alpha=0.5, color=colors[i])
      for data_point in data_points:
         ax.plot(data_point, 0, marker='o', markersize=4, color=colors[i])
      ax.axvline(x=np.percentile(data_points, 25),linestyle='dotted',color=colors[i], label='25th percentile ('+"{0:.2f}".format(np.percentile(data_points, 25))+')')
      ax.axvline(x=np.median(data_points),linestyle='dashed',color=colors[i], label='Median ('+"{0:.2f}".format(np.median(data_points))+')')
      ax.axvline(x=np.mean(data_points),linestyle='dashdot',color=colors[i], label='Mean ('+"{0:.2f}".format(np.mean(data_points))+')')

   ax.legend(loc=1)
   fig.text(0.5, 0.0, 'Belief of inconsistent tuples', ha='center')
   fig.text(0.0, 0.5, 'Probability density', va='center', rotation='vertical')

   fig.tight_layout()
   fig.savefig(inconsistency_out_file_prefix+'.distribution_belief_of_inconsistencies.pdf')
   plt.close()

def report_inconsistency_per_source(pd_data, inconsistencies):
   pd_data_copy = pd_data.copy()
   pd_data_copy['Inconsistency'] = 'No'
   
   for inconsistency_tuples in inconsistencies.values():
      for inconsistency_tuple, sources in inconsistency_tuples:
         for source in sources:
            pd_inconsistency_tuple = pd.Series(inconsistency_tuple + (source,),index=COLUMN_NAMES)
            matched_indices = (pd_data[COLUMN_NAMES] == pd_inconsistency_tuple).all(1)
            if matched_indices.any():
               found_tuple = pd_data[matched_indices]
               pd_data_copy.at[found_tuple.index.values[0],'Inconsistency'] = 'Yes'

   pd_source_size_data  = pd_data_copy.groupby('Source').size()
   inconsistency_ratios = []
   for source in pd_source_size_data.index.tolist():
      search_keyword = pd.Series([source, 'Yes'], index = ['Source', 'Inconsistency'])
      num_of_inconsistencies = sum((pd_data_copy[['Source', 'Inconsistency']] == search_keyword).all(1)) 
      print("[inconsistency ratio] {} {} {}".format(num_of_inconsistencies, pd_source_size_data[source], float(num_of_inconsistencies) / pd_source_size_data[source]))
      inconsistency_ratios.append(float(num_of_inconsistencies) / pd_source_size_data[source])

   return pd_data_copy, (np.mean(inconsistency_ratios), np.std(inconsistency_ratios))

def generate_sankey_data(pd_data, inconsistencies):
   pd_data_copy, inconsistency_stat = report_inconsistency_per_source(pd_data, inconsistencies)

   pd_grouped_data = pd_data_copy.groupby(['Source', 'Predicate', 'Inconsistency']).size()
   #pd_grouped_data = pd.DataFrame(pd_data_copy.groupby(['Source', 'Predicate', 'Inconsistency']).size())
   #pd_grouped_data.at[pd_grouped_data['Inconsistency'] == 'Yes','Inconsistency'] = 'Inconsistent tuples ('+str(len(inconsistencies))+'unique tuples)'
   #pd_grouepd_data.at[pd_grouped_data['Inconsistency'] == 'No','Inconsistency'] = 'Consistent tuples ('+str(pd_data.shape[0])+'unique tuples)'
   out_sankey_data = open('sankey_data_summary.txt','w')

   out_sankey_data.write(','.join(['source','type','target','value'])+'\n')
   for index in pd_grouped_data.index:
      out_sankey_data.write("{},{},{},{}\n".format(index[0],index[1],index[2],pd_grouped_data.loc[index]))

def save_resolved_inconsistencies(inconsistency_out_file, inconsistencies_with_max_belief):
   inconsistency_out = open(inconsistency_out_file, 'w')

   inconsistency_out.write('{}\n'.format('\t'.join(['Subject','Predicate','Object','Belief','Source size','Sources','Total source size','Conflicting tuple info'])))
   for inconsistent_tuples_with_max_belief in inconsistencies_with_max_belief.values():
      (selected_tuple, sources, belief) = inconsistent_tuples_with_max_belief[0]
      conflicting_tuple_info            = inconsistent_tuples_with_max_belief[1:]

      total_source_size = np.sum([len(inconsistent_tuple_with_max_belief[1]) for inconsistent_tuple_with_max_belief in inconsistent_tuples_with_max_belief])

      print('[inconsistency resolution] {}\t{}\t{}\t{}\t{}\t{}'.format('\t'.join(selected_tuple),
                                                         "{0:.10f}".format(belief),
                                                         len(sources),
                                                         ",".join(sources),
                                                         total_source_size,
                                                         conflicting_tuple_info))
      inconsistency_out.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('\t'.join(selected_tuple),
                                                         "{0:.10f}".format(belief),
                                                         len(sources),
                                                         ",".join(sources),
                                                         total_source_size,
                                                         conflicting_tuple_info))

   inconsistency_out.close()

def save_integrated_data(data_out_file, pd_belief_vector_without_inconsistencies):
   data_out = open(data_out_file, 'w')

   data_out.write('\t'.join(['Subject','Predicate','Object','Belief','Source size','Sources'])+'\n')
   for tuple, belief_and_sources in pd_belief_vector_without_inconsistencies.iterrows():
      data_out.write('\t'.join(tuple)+'\t'+str("{0:.2f}".format(belief_and_sources[0]))+'\t'+str(len(belief_and_sources[1]))+'\t'+','.join(belief_and_sources[1])+'\n')

   data_out.close()