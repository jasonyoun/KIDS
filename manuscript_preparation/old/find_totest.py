import sys
import pandas as pd

########
# read #
########
pd_data = pd.read_csv(
	'./hypotheses_confidence.txt',
	names=['Subject', 'Predicate', 'Object', 'Label', 'Probability'],
	sep='\t')

antibiotics = [
	'Sisomicin',
	'Paromomycin',
	'Apramycin',
	'Hygromycin B',
	'Metronidazole',
	'Triclosan',
	'Ampicillin',
	'Geneticin',
	'Troleandomycin',
	'Rifampin',
	'Cephradine',
	'Spectinomycin',
	'Levofloxacin',
	'Chloramphenicol',
	'Streptomycin',
	'Novobiocin',
	'Norfloxacin',
	'Kanamycin',
	'Vancomycin',
	'Amoxicillin',]

##########
# sample #
##########

list_selected = []

pd_data_above10 = pd_data[pd_data['Probability'] >= 0.1]

for antibiotic in antibiotics:
	pd_selected = pd_data_above10[pd_data_above10['Object'] == antibiotic]
	list_selected.append(pd_selected)

pd_totest = pd.concat(list_selected, sort=True)

pd_totest.to_csv('~/Jason/UbuntuShare/totest.txt', sep='\t', index=False)
