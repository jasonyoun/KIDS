import pandas as pd

pd_new = pd.read_csv('./totest_new.txt', sep='\t')
pd_old = pd.read_csv('./totest_old.txt', sep='\t')

pd_duplicated = pd.concat([pd_new, pd_old], axis=0, ignore_index=True)
index_duplicated = pd_duplicated.duplicated(subset=['Subject', 'Predicate', 'Object'])
pd_duplicated = pd_duplicated[index_duplicated].reset_index(drop=True)
print(pd_duplicated.shape)

pd_totest = pd.concat([pd_new, pd_duplicated], axis=0, ignore_index=True)
index_duplicated = pd_totest.duplicated(subset=['Subject', 'Predicate', 'Object'], keep=False)
pd_totest = pd_totest[~index_duplicated].reset_index(drop=True)

pd_totest.to_csv('~/Jason/UbuntuShare/totest.txt', sep='\t', index=False)
