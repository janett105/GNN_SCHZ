import numpy as np
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
    input : 각 site(batch)의 모든 FC data array
    output : site effect가 통계적으로 유의미하게 존재하는지 여부

    HC_site1,SCZ_2 : HC의 
    SCZ_site1,SCZ_2 : SCZ의 각 site(batch)의 모든 FC data array
    """
    # Kruskal-Wallis 검정 수행
    stat, p_value = kruskal(site1, site2)
    print(f"Kruskal-Wallis 검정 통계량: {stat}, p-값: {p_value}")

    p_values_adjusted = multipletests([p_value], alpha=0.05, method='fdr_bh')[1]
    print("FDR 보정된 p-값:", p_values_adjusted[0])
    
    if p_values_adjusted<.05:
        return True
    else:
        return False