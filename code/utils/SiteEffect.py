import numpy as np
import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

def HC_SCZ_SiteEffectExists(x, labels, batch):
    """
    HC_site1,SCZ_2 : HC의 
    SCZ_site1,SCZ_2 : SCZ의 각 site(batch)의 모든 FC data array
    """
    

    SiteEffectExists_Kruskal(HC_site1, HC_site2)
    SiteEffectExists_Kruskal(SCZ_site1, SCZ_site2)



def SiteEffectExists_Kruskal(site1, site2):
    """
    input : 각 site(batch)의 HC/SCZ FC data array
    output : site effect가 통계적으로 유의미하게 존재하는지 여부, 그 분포
    """
    data = {
    'site1': site1,
    'site2': site2
    }
    df = pd.DataFrame(data)

    # Kruskal-Wallis 검정 수행 및 p-value 저장
    p_values = []
    for site in df.columns:
        # 각 기능적 연결성에 대해 모든 사이트의 데이터를 사용하여 Kruskal-Wallis 검정 수행
        _, p = kruskal(*[df[site] for site in df.columns])
        p_values.append(p)
    


# 예제 데이터 생성: 여기서는 각 사이트(site1, site2, site3)에서 얻은 기능적 연결성 데이터를 간단히 시뮬레이션합니다.
np.random.seed(0)  # 결과의 일관성을 위해
data = {
    'site1': np.random.normal(loc=0, scale=1, size=100),
    'site2': np.random.normal(loc=0.5, scale=1, size=100),
    'site3': np.random.normal(loc=-0.5, scale=1, size=100)
}

# DataFrame으로 변환
df = pd.DataFrame(data)

# Kruskal-Wallis 검정 수행 및 p-value 저장
p_values = []
for connectivity in df.columns:
    # 각 기능적 연결성에 대해 모든 사이트의 데이터를 사용하여 Kruskal-Wallis 검정 수행
    _, p = kruskal(*[df[site] for site in df.columns]) # *는 리스트의 각 항목을 별도의 인자로 함수 전달 
    p_values.append(p)

# FDR 보정 적용
_, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# 유의미한 사이트 효과가 있는 연결성 확인
significant_effects = p_corrected < 0.05

# 결과 출력
for i, connectivity in enumerate(df.columns):
    print(f"{connectivity}: {'유의미한 사이트 효과 있음' if significant_effects[i] else '유의미한 사이트 효과 없음'}")

# 유의미한 사이트 효과가 있는 연결성의 비율 계산 및 출력
print(f"\n유의미한 사이트 효과가 있는 연결성의 비율: {np.mean(significant_effects) * 100:.2f}%")
