"""
Filename: report_manager.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu

Description:
    Functions for generating a report.

To-do:
    1. put function comments and do a bit of a cleanup
"""
#!/usr/bin/python

# import generic packages
import math
import logging as log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# import from knowledge_scholar
from .utilities import get_belief_of_inconsistencies

# set fonts
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# constants
SPO_LIST = ['Subject', 'Predicate', 'Object']
COLUMN_NAMES = SPO_LIST + ['Source']

# IT IS NOT GENERIC (NEED TO UPDATE)
DATA_CATEGORY = {'Soo'    : 'Phenotype microarray',
                 'Shaw'   : 'Expression profile',
                 'Zhou'   : 'Phenotype microarray',
                 'Nichols': 'Growth profile',
                 'Liu'    : 'MIC profile',
                 'Tamae'  : 'MIC profile',
                 'Girgis' : 'Transposon',
                 'CARD'   : 'KB',
                 'GO'     : 'KB',
                 'hiTRN'  : 'KB'}

def plot_network_of_inconsistency(pd_data, inconsistencies):
    pd_grouped_data = pd_data.groupby('Source').apply(len)
    pd_edges = pd.DataFrame(columns=['source', 'target'])
    pd_edges_idx = 0
    source_to_target = {}

    for inconsistent_tuples in inconsistencies.values():
        inconsistent_sources = []
        for inconsistent_tuple, sources in inconsistent_tuples:
            for source in sources:
                inconsistent_sources.append(source)
        source = inconsistent_sources[0]
        for target in inconsistent_sources[1:]:
            if target in source_to_target and source == source_to_target[target]:
                pd_edges.loc[pd_edges_idx] = [target, source]
            else:
                pd_edges.loc[pd_edges_idx] = [source, target]
            pd_edges_idx = pd_edges_idx + 1

    pd_grouped_edges = pd.DataFrame(pd_edges.groupby(['source', 'target']).size())

    #node_out = open('inconsistency_network.nodes.txt','w')
    #edge_out = open('inconsistency_network.edges.txt','w')

    #node_out.write("Id\tSize\tType\n")
    #for i in range(len(pd_grouped_data)):
    #   node_out.write("{}\t{}\t{}\n".format(pd_grouped_data.index[i], math.log2(pd_grouped_data.values[i]), DATA_CATEGORY[pd_grouped_data.index[i]]))

    #edge_out.write("Source\tTarget\tWeight\n")
    #for i in range(len(pd_grouped_edges)):
    #   edge_out.write("{}\t{}\t{}\n".format(pd_grouped_edges.index[i][0], pd_grouped_edges.index[i][1], math.log2(pd_grouped_edges.values[i])))

def plot_pie_summary(pd_data, inconsistencies):
    inconsistency_source_category = pd.Series(index=inconsistencies.keys())
    """
    for inconsistency_tuples in inconsistencies.values():
        for inconsistency_tuple, sources in inconsistency_tuples:
            for source in sources:
                inconsistency_source_category DATA_CATEGORY[source]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)

    plt.axis('equal')
    plt.savefig('integrated_data_summary.pdf')
    plt.close()
    """

def get_inconsistencies_of_source(source, inconsistencies):
    num_inconsistencies = 0

    for inconsistent_tuples in inconsistencies.values():
        for _, sources in inconsistent_tuples:
            if source in sources:
                num_inconsistencies += 1

    return num_inconsistencies

def plot_trustworthiness(pd_data, np_trustworthiness_vector, inconsistencies):
    pd_data_copy = pd_data.copy()

    pd_grouped_data = pd_data_copy.groupby(SPO_LIST)['Source'].apply(set)
    sources = pd_data.groupby('Source').size().index.tolist()
    list_trustworthiness_vector = [float(trustworthiness) for trustworthiness in np_trustworthiness_vector]
    pd_trustworthiness_vector = pd.Series(list_trustworthiness_vector, index=sources)

    pd_data_stat_column_names = ['Single source', 'Multiple sources', 'Inconsistencies']
    pd_data_stat = pd.DataFrame(index=sources, columns=pd_data_stat_column_names)

    for source in sources:
        num_of_tuples_with_one_source = sum(pd_grouped_data == {source})
        num_of_rest = sum(pd_data_copy['Source'] == source) - num_of_tuples_with_one_source
        pd_data_stat.loc[source] = [num_of_tuples_with_one_source, num_of_rest, get_inconsistencies_of_source(source, inconsistencies)]

    _, ax1 = plt.subplots()
    ax1.set_yscale("log")

    dim = len(pd_data_stat_column_names)
    w = 0.5
    dimw = w / dim

    sorted_sources = pd_trustworthiness_vector.sort_values(ascending=False).index.tolist()
    pd_data_stat = pd_data_stat.loc[sorted_sources]

    log.debug(pd_data_stat)
    log.debug(pd_trustworthiness_vector)

    x = np.arange(len(sources))
    i = -1
    for column_name in pd_data_stat_column_names:
        ax1.bar(x + i * dimw, pd_data_stat[column_name], dimw, label=column_name, bottom=0.001)
        i += 1
    #ax2 = ax1.twinx()
    #ax2.plot(x, pd_trustworthiness_vector[sorted_sources].tolist(), marker = 'o', color = 'r', label = 'Trustworthiness')
    #ax1.plot(np.nan, marker = 'o', color = 'r', label = 'Trustworthiness')

    ax1.set_xlabel('Sources')
    ax1.set_ylabel('Number of tuples')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_sources)

    #ax2.set_ylabel('Relative trustworthiness')
    #ax2.set_xticks(x)
    ax1.legend()
    plt.savefig('./output/trustworthiness_data_summary.pdf')
    plt.close()

def plot_belief_of_inconsistencies(inconsistencies_with_max_belief, answer, inconsistency_out_file_prefix):
    data = get_belief_of_inconsistencies(inconsistencies_with_max_belief, answer, SPO_LIST)
    colors = ['b', 'g']
    labels = ['Correctly resolved tuple', 'Incorrectly resolved tuple']

    fig, axes = plt.subplots()
    axes.set_xlim(0, 1)
    for i in data:
        data_points = data[i]

        axes.hist(data_points, normed=True, label=labels[i], histtype='stepfilled', alpha=0.5, color=colors[i])
        for data_point in data_points:
            axes.plot(data_point, 0, marker='o', markersize=4, color=colors[i])
        axes.axvline(x=np.percentile(data_points, 25), linestyle='dotted', color=colors[i], label='25th percentile ('+"{0:.2f}".format(np.percentile(data_points, 25))+')')
        axes.axvline(x=np.median(data_points), linestyle='dashed', color=colors[i], label='Median ('+"{0:.2f}".format(np.median(data_points))+')')
        axes.axvline(x=np.mean(data_points), linestyle='dashdot', color=colors[i], label='Mean ('+"{0:.2f}".format(np.mean(data_points))+')')

    axes.legend(loc=1)
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
                pd_inconsistency_tuple = pd.Series(inconsistency_tuple + (source,), index=COLUMN_NAMES)
                matched_indices = (pd_data[COLUMN_NAMES] == pd_inconsistency_tuple).all(1)
                if matched_indices.any():
                    found_tuple = pd_data[matched_indices]
                    pd_data_copy.at[found_tuple.index.values[0], 'Inconsistency'] = 'Yes'

    pd_source_size_data = pd_data_copy.groupby('Source').size()
    inconsistency_ratios = []
    for source in pd_source_size_data.index.tolist():
        search_keyword = pd.Series([source, 'Yes'], index=['Source', 'Inconsistency'])
        num_of_inconsistencies = sum((pd_data_copy[['Source', 'Inconsistency']] == search_keyword).all(1))
        log.debug('%d %d %f', num_of_inconsistencies, pd_source_size_data[source], float(num_of_inconsistencies) / pd_source_size_data[source])
        inconsistency_ratios.append(float(num_of_inconsistencies) / pd_source_size_data[source])

    return pd_data_copy, (np.mean(inconsistency_ratios), np.std(inconsistency_ratios))

def generate_sankey_data(pd_data, inconsistencies):
    pd_data_copy, _ = report_inconsistency_per_source(pd_data, inconsistencies)

    pd_grouped_data = pd_data_copy.groupby(['Source', 'Predicate', 'Inconsistency']).size()
    #pd_grouped_data = pd.DataFrame(pd_data_copy.groupby(['Source', 'Predicate', 'Inconsistency']).size())
    #pd_grouped_data.at[pd_grouped_data['Inconsistency'] == 'Yes','Inconsistency'] = 'Inconsistent tuples ('+str(len(inconsistencies))+'unique tuples)'
    #pd_grouepd_data.at[pd_grouped_data['Inconsistency'] == 'No','Inconsistency'] = 'Consistent tuples ('+str(pd_data.shape[0])+'unique tuples)'
    out_sankey_data = open('sankey_data_summary.txt', 'w')

    out_sankey_data.write(','.join(['source', 'type', 'target', 'value']) + '\n')
    for index in pd_grouped_data.index:
        out_sankey_data.write("{},{},{},{}\n".format(index[0], index[1], index[2], pd_grouped_data.loc[index]))
