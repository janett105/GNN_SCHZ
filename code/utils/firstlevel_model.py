from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import clean_img, load_img
import pandas as pd

def first_level_model(subj_id, subj_dir):
    fmri_img_path = f'{subj_dir}{subj_id}/func/{subj_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    confounds = pd.read_csv(f'{subjects_dir}{subj_id}/func/{subj_id}_rest_confounds.csv').values

    first_level_model = FirstLevelModel(t_r=2.0,
                                        high_pass=.01,  # 고주파 필터링 값 설정
                                        smoothing_fwhm=5,  # 이미지 스무딩 값 설정
                                        )

    # GLM 적용 및 confound 변수 제거
    first_level_model = first_level_model.fit(fmri_img_path, confounds=confounds)

    # # 통계적 매개변수 맵(statistical parameter maps, SPMs) 얻기
    # # 'effects'나 'z_score'와 같은 통계적 지표를 사용할 수 있음
    # z_map = first_level_model.compute_contrast('your_contrast_definition', output_type='z_score')

    # # 결과 저장
    # output_path = 'path/to/your/output_z_map.nii.gz'  # 출력 파일 경로
    # z_map.to_filename(output_path)

    # # 위 코드는 confound 변수를 제거하고 GLM을 적용한 후 특정 대조(contrast)에 대한 z-스코어 맵을 생성합니다.
    # # 'your_contrast_definition'에는 대조를 정의하는 데 사용할 설계 행렬의 열 이름이나 대조 벡터가 들어갑니다.

subjects_dir = 'D:/MRI/UCLA_CNP/preprocess/fmriprep/derivatives/'