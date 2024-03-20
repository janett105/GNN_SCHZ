# cd /home/jihoo/data/UCLA_CNP 먼저 수행
# scaling : blur처리한 영상을 blur영상의 평균영상으로 나눠서 정규화(백분율로 변환), 
#           해당 계산에서 각 voxel값이 200을 넘지 않도록 하고, 
#           blur영상과 blur평균 영상 모두 모든 voxel의 값이 양수일 경우에만 계산을 진행
#           해당 계산이 brain mask에 해당하는 부분에서만 진행되도록 함

for subjid in 10159; do
    cd sub-${subjid}/func
    3dmerge -1blur_fwhm 4.0 -doall -prefix r${subjid}_blur.nii \
        sub-${subjid}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    3dTstat -prefix rm.mean_r${subjid}.nii r${subjid}_blur.nii
    3dcalc -a r${subjid}_blur.nii -b rm.mean_r${subjid}.nii \
        -c sub-${subjid}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz \
        -expr 'c * min(200, a/b*100)*step(a)*step(b)' \
        -prefix r${subjid}_scale.nii
    rm rm*
    cd ../../
done