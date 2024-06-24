import glob
import pandas as pd

original_df = pd.read_csv('Z:/MRI/DecNef/origin/SRPBS_1600/phenotype_SCZ_HC.csv')
original_subjects_set = set(original_df.iloc[:, 0])

FCfile = glob.glob('Z:/MRI/DecNef/processed/fmriprep/*')
FClst=[]

print(FCfile[10])
for word in FCfile:
    temp = word.replace('Z:/MRI/DecNef/processed/fmriprep\\','')
    FClst.append(temp)
print(FClst[10])
a=set(FClst)

print(len(original_subjects_set))
print(len(a))
print(len(original_subjects_set-a))
print(list(original_subjects_set-a))