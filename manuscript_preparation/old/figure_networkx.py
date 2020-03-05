import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from nxviz.plots import CircosPlot
import pandas as pd
import itertools

entity_list = []
with open('./entity_full_names.txt') as file:
	content = file.readlines()
	for line in content:
		items = line.strip().split(':', maxsplit=2)
		entity_list.append([items[1], items[2]])

pd_entities = pd.DataFrame(entity_list, columns=['entity_type', 'entity_name'])
genes = pd_entities[pd_entities['entity_type'] == 'gene']['entity_name'].to_numpy().tolist()
antibiotics = pd_entities[pd_entities['entity_type'] == 'antibiotic']['entity_name'].to_numpy().tolist()

empty_data_list = []
for gene_antibiotic_pair in itertools.product(genes, antibiotics):
	empty_data_list.append(list(gene_antibiotic_pair))

pd_empty = pd.DataFrame(empty_data_list, columns=['Subject', 'Object'])
pd_empty['Count'] = 0

pd_hypotheses = pd.read_csv('./hypotheses.txt', sep='\t', names=['Subject', 'Predicate', 'Object', 'Label'])
pd_hypotheses = pd_hypotheses[['Subject', 'Object']]
pd_hypotheses['Count'] = 0.05

pd_to_graph = pd.concat([pd_empty, pd_hypotheses]).groupby(['Subject', 'Object']).sum().reset_index()

G = nx.from_pandas_edgelist(pd_to_graph, source='Subject', target='Object', edge_attr=True)

genes = np.unique(pd_to_graph['Subject'].to_numpy())
antibiotics = np.unique(pd_to_graph['Object'].to_numpy())

for n in G.nodes():
	if n in genes:
		G.node[n]['Type'] = 0
	elif n in antibiotics:
		G.node[n]['Type'] = 1
	else:
		raise ValueError('Invalid node {}'.format(n))

c = CircosPlot(
    G,
    dpi=600,
    node_grouping='Type',
    edge_width='Count',
    figsize=(20, 20),
    node_color='Type',
)

c.draw()
plt.show()
