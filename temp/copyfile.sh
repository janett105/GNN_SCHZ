#!/bin/sh
for subid in {40002} #40147
do
# 폴더 시스템 만들기
mkdir sub-{$subid}
cd sub-{$subid}
mkdir anat
mkdir func

# 각 폴더에 복붙
cp home/NAS/COBRE/origin/functional/COBRE/00{$subid}/session_1/anat_1/mprage.nii.gz home/jihoo/COBRE/subject/sub-{$subid}/anat
cd anat
mv mprage.nii.gz sub-{$subid}_T1w.nii.gz
cd ../func
cp home/NAS/COBRE/origin/functional/COBRE/00{$subid}/session_1/rest_1/rest.nii.gz home/jihoo/COBRE/subject/sub-{$subid}/func
mv rest.nii.gz sub-{$subid}_task-rest_bold.nii.gz

# json 파일 모든 폴더에 복붙 
cp home/NAS/COBRE/origin/functional/COBRE/COBRE_parameters_rest.json home/jihoo/COBRE/subject/sub-{$subid}/func
mv COBRE_parameters_rest.json sub-{$subid}_task-rest_bold.json
done

