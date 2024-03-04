#!/bin/bash

fmriprep_run() {
    docker run -it --rm \
    -v /home/jihoo/COBRE/subject:/data \
    -v /home/jihoo/COBRE/:/homedir \
    -v /home/jihoo/COBRE/subject/derivatives:/out \
    nipreps/fmriprep:22.0.2 \
    /data /out participant \
    --fs-license-file /homedir/license.txt \
    --mem 30000 \
    --nthreads 20
}
# 91~100 세팅 완료 후 실행
old_start=91
old_end=100

fmriprep_run
while [ $? -eq 0 ]; do
    new_start={$old_start}+10 #101
    new_start={$old_start}+19  #110

    # fmri prep에 맞게 변경한 original data 구조 저장 후 삭제
    cd /home/jihoo/COBRE/subject && \
    cp -r sub-{{$old_subjs}..{}} /home/jihoo/UCLA_CNP_FC/COBRE/original/ && \
    rm -r sub-{$old_subjs} && \

    # fmri prep 전처리 결과 저장 후 삭제
    cd derivatives
    cp -r sub-{$old_subjs} /home/jihoo/UCLA_CNP_FC/COBRE/derivatives/ && \
    cp -r *.html /home/jihoo/UCLA_CNP_FC/COBRE/derivatives/ && \
    rm -r sub-{$old_subjs} && \
    rm *.html && \

    # 다음 데이터 10개를 frmi-prep에 맞게 구조 변경하고 가져옴
    cd ../ && \
    cp /home/jihoo/UCLA_CNP_FC/COBRE/copyfile.sh /home/jihoo/COBRE/subject/ && \
    sed -i 's/ids={$old_subjs}/VALUE={$new_subjs}/' copyfile.sh && \
    sh copyfile.sh && \
    rm -r copyfile.sh && \

    fmriprep_run
done

echo "Error occurred during preprocessing"
exit 1