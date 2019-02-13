from __future__ import division, absolute_import, print_function
import os
import sys
import pandas as pd
from pylab import *
from numpy.random import *
from graph_tool.all import *
from postprocess_modules.extract_info import ExtractInfo

# predicates
CRTA_STR = 'confers resistance to antibiotic'
CNRTA_STR = 'confers no resistance to antibiotic'
UBA_STR = 'upregulated by antibiotic'
NUBA_STR = 'not upregulated by antibiotic'
R_STR = 'represses'
NR_STR = 'no represses'
A_STR = 'activates'
NA_STR = 'no activates'
HAS_STR = 'has'
III_STR = 'is involved in'
IPO_STR = 'is part of'
TB_STR = 'targeted by'

# get knowledge graph and process it
# pd_data = pd.read_csv('/home/jyoun/Jason/Scratch/dev.txt', sep='\t', names=['Subject', 'Predicate', 'Object', 'Label'])
# pd_data = pd_data[pd_data['Label'].astype(str) == '1'].reset_index(drop=True)

pd_data = pd.read_csv('./output/data.txt', sep='\t', names=['Subject', 'Predicate', 'Object', 'Label'])
pd_data = pd_data.sample(frac=1).reset_index(drop=True)
pd_data = pd_data.iloc[0:2500, :]

# separate dataset into entities and relations
ei = ExtractInfo(pd_data, './data/domain_range.txt')
genes = ei.get_entity_by_type('gene')
antibiotics = ei.get_entity_by_type('antibiotic')
molecular_functions = ei.get_entity_by_type('molecular_function')
biological_processes = ei.get_entity_by_type('biological_process')
cellular_components = ei.get_entity_by_type('cellular_component')

# combine all into dictionary
entities_dict = {}

if genes.shape[0] != 0: entities_dict['gene'] = genes.tolist()
if antibiotics.shape[0] != 0: entities_dict['antibiotic'] = antibiotics.tolist()
if molecular_functions.shape[0] != 0: entities_dict['molecular_function'] = molecular_functions.tolist()
if biological_processes.shape[0] != 0: entities_dict['biological_process'] = biological_processes.tolist()
if cellular_components.shape[0] != 0: entities_dict['cellular_component'] = cellular_components.tolist()

# start constructing graph
g = Graph(directed=False)

vprop_int = g.new_vertex_property('int')
vprop_str = g.new_vertex_property('string')
eprop_int = g.new_edge_property('int')
eprop_str = g.new_edge_property('string')

vdic = {}

for key, val in entities_dict.items():

	if key == 'gene':
		int_prop = 0
		str_prop = 'g'
	elif key == 'antibiotic':
		int_prop = 1
		str_prop = 'a'
	elif key == 'molecular_function':
		int_prop = 2
		str_prop = 'm'
	elif key == 'biological_process':
		int_prop = 3
		str_prop = 'b'
	elif key == 'cellular_component':
		int_prop = 4
		str_prop = 'c'
	else:
		sys.exit('something wrong')

	for i in range(len(val)):
		v = g.add_vertex()
		vprop_int[v] = int_prop
		vprop_str[v] = str_prop
		vdic[val[i]] = v

for i in range(pd_data.shape[0]):
	triple = pd_data.iloc[i, :]
	sub = triple.Subject
	pred = triple.Predicate
	obj = triple.Object

	# add edge
	e = g.add_edge(vdic[sub], vdic[obj])

	if pred == CRTA_STR or pred == CNRTA_STR:
		int_prop = 0
		str_prop = 'c'
	elif pred == UBA_STR or pred == NUBA_STR:
		int_prop = 1
		str_prop = 'u'
	elif pred == R_STR or pred == NR_STR:
		int_prop = 2
		str_prop = 'r'
	elif pred == A_STR or pred == NA_STR:
		int_prop = 3
		str_prop = 'a'
	elif pred == HAS_STR:
		int_prop = 4
		str_prop = 'h'
	elif pred == III_STR:
		int_prop = 5
		str_prop = 'i'
	elif pred == IPO_STR:
		int_prop = 6
		str_prop = 'p'
	elif pred == TB_STR:
		int_prop = 7
		str_prop = 't'
	else:
		sys.exit('something wrong')

	eprop_int[e] = int_prop
	eprop_str[e] = str_prop

# # save graph
# print('saving graph...')
# g.vertex_properties["kbase"] = vprop_int
# g.edge_properties["kbase"] = eprop_int
# g.save("kbase.xml.gz")

# # load graph
# print('loading graph...')
# g = load_graph("kbase.xml.gz")
# vprop = g.vertex_properties["kbase"]
# eprop = g.vertex_properties["kbase"]

# draw graph
print('drawing graph...')

pos = arf_layout(g)
# pos = arf_layout(g, max_iter=150, dt=1e-5, dim=3)

# deg = g.degree_property_map("in")
# deg.a = 4 * (sqrt(deg.a) * 0.5 + 0.4)
# ebet = betweenness(g)[1]
# ebet.a /= ebet.a.max() / 10.
# eorder = ebet.copy()
# eorder.a *= -1

# graph_draw(
# 	g,
# 	pos,
# 	output_size=(1000, 1000),
# 	vertex_fill_color=deg,
# 	vorder=deg,
# 	edge_pen_width=0.5,
# 	edge_color=[0, 0, 0, 0.3],
# 	eorder=eorder,
# 	output="kbase.png")

graph_draw(
	g,
	pos,
	output_size=(1000, 1000),
	vertex_color=[1,1,1,0],
	vertex_fill_color=vprop_int,
	# vertex_text=vprop_str,
	vertex_size=6,
	vcmap=matplotlib.cm.tab20c,
	edge_pen_width=1.5,
	edge_color=eprop_int,
	# edge_text=eprop_str,
	ecmap=matplotlib.cm.Set2,
	output="kbase.png")
