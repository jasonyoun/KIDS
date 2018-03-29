import random
with open('hiTRN_KG_without_multi-gene-TFs.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('scrambled.txt','w') as target:
    for _, line in data:
        target.write( line )