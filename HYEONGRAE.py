# 데이터 프레임 만들고 '열'의 평균 각각 구하기 

from sklearn.mixture import GaussianMixture
import pandas as pd

#[정상 데이터] 
normal = pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/feature_normal.xls')
'''
X = df.filter(items=['IA_am','IA_bm','IA_cm','IA_aph','IA_bph','IA_cph','IB_am','IB_bm','IB_cm','IB_aph','IB_bph','IB_cph','target','type'])
'''
mean_VA_am = normal['VA_am'].mean()
mean_VA_bm = normal['VA_bm'].mean()
mean_VA_cm = normal['VA_cm'].mean()
mean_VA_aph = normal['VA_aph'].mean()
mean_VA_bph = normal['VA_bph'].mean()
mean_VA_cph = normal['VA_cph'].mean()

mean_IA_am = normal['IA_am'].mean()
mean_IA_bm = normal['IA_bm'].mean()
mean_IA_cm = normal['IA_cm'].mean()
mean_IA_aph = normal['VA_aph'].mean()
mean_IA_bph = normal['VA_bph'].mean()
mean_IA_cph = normal['VA_cph'].mean()
 




print()