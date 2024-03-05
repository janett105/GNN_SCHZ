import glob
import pandas as pd

# 기존 phenotype에서 fmri-prep + FC 생성 가능한 subject 리스트 생성
original_df = pd.read_csv('phenotype.csv')
c= original_df.iloc[:, 0]

FCfile = glob.glob('*.npy')
filelst=[]
print(filelst)
for word in FCfile:
    temp = word.replace('data/raw\\FC116_','')
    temp = temp.replace('.npy', '')
    filelst.append(temp)
FC_df=pd.DataFrame(filelst, columns=['participant_id'])

new_df = pd.merge(original_df, FC_df, how='inner', on='participant_id')

new_df['Dataset'] = 'COBRE'

new_df.to_csv('Labels_164parcels.csv', index=False)