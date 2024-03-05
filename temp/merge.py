import glob
import pandas as pd

whole = pd.read_csv('data/raw/phenotype.csv')
c= whole['participant_id']

a = glob.glob('data/raw/*.npy')
b=[]
for word in a:
    temp = word.replace('data/raw\\FC116_','')
    temp = temp.replace('.npy', '')
    b.append(temp)
B=pd.DataFrame(b, columns=['participant_id'])

d = pd.merge(whole, B, how='inner', on='participant_id')

d.to_csv('data/raw/Labels_116parcels.csv', index=False)