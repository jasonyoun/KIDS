import sys
import pandas as pd
import numpy as np

# read
pd_hypotheses_1 = pd.read_csv(
	'./hypotheses_confidence.txt',
	names=['Subject', 'Predicate', 'Object', 'Label', 'Probability'],
	sep='\t')
pd_hypotheses_1 = pd_hypotheses_1[['Subject', 'Predicate', 'Object', 'Probability']]

pd_hypotheses_2 = pd.read_csv(
	'./hypotheses_confidence_2.txt',
	names=['Subject', 'Predicate', 'Object', 'Label', 'Probability'],
	sep='\t')
pd_hypotheses_2 = pd_hypotheses_2[['Subject', 'Predicate', 'Object', 'Probability']]

pd_validated_hypotheses_1 = pd.read_csv(
	'./validated_hypotheses_1.txt',
	sep='\t')

antibiotics_in_stock = list(set(pd_validated_hypotheses_1['Object'].tolist()))

# merge
pd_hypotheses_merged = pd_hypotheses_1.merge(
	pd_hypotheses_2,
	how='outer',
	on=['Subject', 'Predicate', 'Object'],
	suffixes=('_1', '_2'))

pd_hypotheses_merged = pd_hypotheses_merged.merge(
	pd_validated_hypotheses_1,
	how='outer',
	on=['Subject', 'Predicate', 'Object'])

pd_hypotheses_merged['Antibiotics in stock'] = pd_hypotheses_merged['Object'].apply(
	lambda x: 'Yes' if x in antibiotics_in_stock else 'No')

pd_hypotheses_merged = pd_hypotheses_merged.sort_values(by=['Probability_2'], ascending=False)
# pd_hypotheses_merged.to_csv('~/Jason/VM_Shared/hypotheses_merged.txt', sep='\t', index=False)


pd_hypotheses_can_test = pd_hypotheses_merged[pd_hypotheses_merged['Antibiotics in stock'] == 'Yes']
pd_hypotheses_over_20 = pd_hypotheses_can_test[pd_hypotheses_can_test['Probability_2'] > 0.2]
pd_hypotheses_below_20 = pd_hypotheses_can_test[pd_hypotheses_can_test['Probability_2'] <= 0.2]

sampled_list = []
for antibiotic_in_stock in antibiotics_in_stock:
	pd_filtered = pd_hypotheses_below_20[pd_hypotheses_below_20['Object'] == antibiotic_in_stock]
	sampled_list.append(pd_filtered.sample(n=2))

pd_hypotheses_sampled_below_20 = pd.concat(sampled_list)

pd_hypotheses_to_test = pd.concat([pd_hypotheses_over_20, pd_hypotheses_sampled_below_20])
pd_hypotheses_to_test.to_csv('~/Jason/VM_Shared/hypotheses_to_test.txt', sep='\t', index=False)

sys.exit()

# test what antibiotics to buy
antibiotics_to_buy = {
	'Rifamycin SV': 15,
	'Cloxacillin': 11,
	'Metronidazole': 9,
	'Cefoxitin': 7,
	'Capreomycin': 6,
	'Colistin': 5,
	'Bleomycin': 4,
	'Fosfomycin': 3,
	'Cefsulodin': 3,
	'Nigericin': 3,
	'Hygromycin ': 3,
	'Lomefloxacin': 3,
	'Doxycycline': 3,
	'Streptonigrin': 3,
	'Doxycycline hyclate': 2,
	'Mecillinam': 2,
	'CHIR090': 2,
	'Mitomycin C': 2,
	'Minocycline': 2,
	'Neomycin': 1,
	'Radicicol': 1,
	'Cefamandole': 1,
	'Clarythromycin': 1,
	'Polymyxin B': 1}


index = (pd_hypotheses_merged['Probability_2'] > 0.8) & (pd_hypotheses_merged['Probability_2'] <= 1.0)
pd_hypotheses_80_100 = pd_hypotheses_merged[index]
yes_no_counts = pd_hypotheses_80_100['Antibiotics in stock'].value_counts()
coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
print('80 100: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_80_100.shape[0])

index = (pd_hypotheses_merged['Probability_2'] > 0.6) & (pd_hypotheses_merged['Probability_2'] <= 0.8)
pd_hypotheses_60_80 = pd_hypotheses_merged[index]
yes_no_counts = pd_hypotheses_60_80['Antibiotics in stock'].value_counts()
coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
print('60 80: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_60_80.shape[0])

index = (pd_hypotheses_merged['Probability_2'] > 0.4) & (pd_hypotheses_merged['Probability_2'] <= 0.6)
pd_hypotheses_40_60 = pd_hypotheses_merged[index]
yes_no_counts = pd_hypotheses_40_60['Antibiotics in stock'].value_counts()
coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
print('40 60: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_40_60.shape[0])

index = (pd_hypotheses_merged['Probability_2'] > 0.2) & (pd_hypotheses_merged['Probability_2'] <= 0.4)
pd_hypotheses_20_40 = pd_hypotheses_merged[index]
yes_no_counts = pd_hypotheses_20_40['Antibiotics in stock'].value_counts()
coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
print('20 40: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_20_40.shape[0])

index = (pd_hypotheses_merged['Probability_2'] >= 0.0) & (pd_hypotheses_merged['Probability_2'] <= 0.2)
pd_hypotheses_00_20 = pd_hypotheses_merged[index]
yes_no_counts = pd_hypotheses_00_20['Antibiotics in stock'].value_counts()
coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
print('0 20: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_00_20.shape[0])
print()

for antibiotic, count in antibiotics_to_buy.items():
	print(antibiotic, count)
	index = (pd_hypotheses_merged['Object'] == antibiotic)
	pd_hypotheses_merged.loc[index, 'Antibiotics in stock'] = 'Yes'

	index = (pd_hypotheses_merged['Probability_2'] > 0.8) & (pd_hypotheses_merged['Probability_2'] <= 1.0)
	pd_hypotheses_80_100 = pd_hypotheses_merged[index]
	yes_no_counts = pd_hypotheses_80_100['Antibiotics in stock'].value_counts()
	coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
	print('80 100: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_80_100.shape[0])

	index = (pd_hypotheses_merged['Probability_2'] > 0.6) & (pd_hypotheses_merged['Probability_2'] <= 0.8)
	pd_hypotheses_60_80 = pd_hypotheses_merged[index]
	yes_no_counts = pd_hypotheses_60_80['Antibiotics in stock'].value_counts()
	coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
	print('60 80: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_60_80.shape[0])

	index = (pd_hypotheses_merged['Probability_2'] > 0.4) & (pd_hypotheses_merged['Probability_2'] <= 0.6)
	pd_hypotheses_40_60 = pd_hypotheses_merged[index]
	yes_no_counts = pd_hypotheses_40_60['Antibiotics in stock'].value_counts()
	coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
	print('40 60: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_40_60.shape[0])

	index = (pd_hypotheses_merged['Probability_2'] > 0.2) & (pd_hypotheses_merged['Probability_2'] <= 0.4)
	pd_hypotheses_20_40 = pd_hypotheses_merged[index]
	yes_no_counts = pd_hypotheses_20_40['Antibiotics in stock'].value_counts()
	coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
	print('20 40: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_20_40.shape[0])

	index = (pd_hypotheses_merged['Probability_2'] >= 0.0) & (pd_hypotheses_merged['Probability_2'] <= 0.2)
	pd_hypotheses_00_20 = pd_hypotheses_merged[index]
	yes_no_counts = pd_hypotheses_00_20['Antibiotics in stock'].value_counts()
	coverage = float(np.round(yes_no_counts['Yes'] * 100 / yes_no_counts.sum(), 0))
	print('0 20: ', coverage, '/ # of yes: ', yes_no_counts['Yes'], '/ count: ', pd_hypotheses_00_20.shape[0])
	print()
