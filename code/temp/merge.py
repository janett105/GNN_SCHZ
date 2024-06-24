import glob
import pandas as pd

# 기존 phenotype에서 fmri-prep + FC 생성 가능한 subject 리스트 생성
original_df = pd.read_csv('D:/MRI/COBRE/original/COBRE_phenotypic_data.csv')
original_subjects_list = original_df.iloc[:, 0]

FCfile = glob.glob('D:/MRI/COBRE/preprocess/fmriprep/FC/Schaefer2018_100parcels_7networks_1mm/*.npy')
FClst=[]

print(FCfile[0])
for word in FCfile:
    temp = word.replace('D:/MRI/COBRE/preprocess/fmriprep/FC/Schaefer2018_100parcels_7networks_1mm\\FC100_sub-','')
    temp = temp.replace('.npy', '')
    FClst.append(int(temp))
print(FClst[0])
FC_df=pd.DataFrame(FClst, columns=['participant_id'])

new_df = pd.merge(original_df, FC_df, how='inner', on='participant_id')

new_df['Cohert'] = 'COBRE'

new_df.to_csv('D:/MRI/COBRE/preprocess/fmriprep/FC/Schaefer2018_100parcels_7networks_1mm/100FC_phenotype.csv', index=False)