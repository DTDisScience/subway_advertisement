# -*- coding: utf-8 -*-
%run my_profile.py

# =============================================================================
# 6호선 제외 데이터 회귀분석
# =============================================================================
# =============================================================================
# 1. 데이터 불러오기
# =============================================================================
station = pd.read_csv('6호선제외_컬럼작업중_230323.csv')

# 1-1) 정거장깊이(m) 컬럼 절대값 산출
#      지상 or 지하 여부에 따른 음수 도출됨
#      깊이라는 측면에서 음수가 의미 없다고 생각됨
station['정거장깊이(m)'] = station['정거장깊이(m)'].map(lambda x : abs(x))

# =============================================================================
# 2. 더미변수 변환
# =============================================================================
from sklearn.preprocessing import LabelEncoder
# 2-1) 형식 컬럼 더미변수 변환
s1 = station_x['형식']
m_le = LabelEncoder()
label_s1 = m_le.fit_transform(s1)
station_x['형식'] = label_s1

# 2-2) 층수 컬럼 더미변수 변환
s2 = station_x['층수']
m_le = LabelEncoder()
label_s2 = m_le.fit_transform(s2)
station_x['층수'] = label_s2

# =============================================================================
# 3. 더미변수 제외 변수 스케일링(StandardScaler)
# =============================================================================
# 3-1) 더미변수 컬럼 분리
station_x_sc_bf = station_x.iloc[:, [0,1,2,3,5,6,7,8,11]]
station_x_dummies = station_x.iloc[:, [4,9,10,12]]

# 3-2) 표준화
from sklearn.preprocessing import StandardScaler as standard
m_sc = standard()
station_x_sc_af = m_sc.fit_transform(station_x_sc_bf)
station_x_sc = DataFrame(data = station_x_sc_af, columns = station_x_sc_bf.columns)

# 3-3) 표준화 컬럼 + 더미변수 컬럼 합치기
station_com = pd.concat([station_x_sc, station_x_dummies], axis = 1)
station_y

# =============================================================================
# 4. Decision Tree
# =============================================================================
from sklearn.tree import DecisionTreeClassifier as dt_c
from sklearn.model_selection import train_test_split
# 4-1) train / test 분리
st_train_x, st_test_x, st_train_y, st_test_y = train_test_split(station_com, station_y, random_state=0)

# 4-2) model 생성 및 학습
m_dt = dt_c()
m_dt.fit(st_train_x, st_train_y)

# 4-3) score 확인
m_dt.score(st_train_x, st_train_y)          # 1.0   << 과적합
m_dt.score(st_test_x, st_test_y)            # 67.92

# 4-4) 변수 중요도 확인
dt_fe_im = m_dt.feature_importances_
# array([0.38622525, 0.099658  , 0.05299961, 0.10078057, 0.01135706,
#        0.2378346 , 0.02749578, 0.04788894, 0.        , 0.01892843,
#        0.        , 0.        , 0.01683175])
DataFrame(data = dt_fe_im.reshape(-1,1), index = station_com.columns)
# 승강장면적(m^2), 형식, 층수 는 변수 중요도 0으로 불필요하다고 판단됨
# 일평균_승하차      0.386225
# 월평균_승하차      0.099658
# 영업일_평균_승하차   0.053000
# 비영업일_평균_승하차  0.100781
# 환승노선수        0.011357
# 스크린수         0.237835
# 포스터수         0.027496
# 대합실면적(m^2)   0.047889
# 승강장면적(m^2)   0.000000
# 정거장깊이(m)     0.018928
# 형식           0.000000
# 층수           0.000000
# 대학여부         0.016832


# =============================================================================
# 5. PCA 진행
# =============================================================================
# 승강장면적(m^2), 형식, 층수 컬럼 삭제
station_com_drop = station_com.drop(['승강장면적(m^2)', '형식', '층수'], axis=1)
from sklearn.decomposition import PCA

# 5-1) PCA를 통한 인공변수 유도
m_pca_1 = PCA(n_components=2)
m_pca_2 = PCA(n_components=3)
m_pca_3 = PCA(n_components=4)
station_com_drop_pca1 = m_pca_1.fit_transform(station_com_drop)      # 유도된 인공변수 값 리턴
station_com_drop_pca2 = m_pca_2.fit_transform(station_com_drop)      # 유도된 인공변수 값 리턴
station_com_drop_pca3 = m_pca_3.fit_transform(station_com_drop)      # 유도된 인공변수 값 리턴
m_pca_1.explained_variance_ratio_.sum()    # 분산설명력 54.67
m_pca_2.explained_variance_ratio_.sum()    # 분산설명력 66.69
m_pca_3.explained_variance_ratio_.sum()    # 분산설명력 76.57

# 5-2) 각 PCA 모델별 유도된 변수의 가중치 확인
m_pca_1.components_
m_pca_2.components_
m_pca_3.components_


# =============================================================================
# 6. 회귀분석
# =============================================================================
# 6-1) 유의성 검정
from statsmodels.formula.api import ols
station_com_drop.columns = Series(station_com_drop.columns).map(lambda x : x.split('(')[0]).map(lambda x : x.replace('_',''))
df_station = pd.concat([station_com_drop, station_y], axis=1)
cols = station_com_drop.columns
for1 = '급지분류' + '~' + '+'.join(cols)
m_ols = ols(formula = for1, data = df_station).fit()
print(m_ols.summary())

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                급지분류   R-squared:                      0.598
# Model:                            OLS   Adj. R-squared:                  0.578
# Method:                 Least Squares   F-statistic:                     29.66
# Date:                Mon, 27 Mar 2023   Prob (F-statistic):           2.21e-34   영가설 기각 따라서 모형 유의함
# Time:                        11:03:08   Log-Likelihood:                -144.77
# No. Observations:                 210   AIC:                             311.5
# Df Residuals:                     199   BIC:                             348.4
# Df Model:                          10                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                   coef     std err         t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept       2.4263      0.040     60.795      0.000       2.348       2.505
# 일평균승하차  1383.0696   2130.732      0.649      0.517   -2818.642    5584.781
# 월평균승하차    -0.0217      0.038     -0.576      0.565      -0.096       0.053
# 영업일승하차 -1076.3775   1658.141     -0.649      0.517   -4346.159    2193.404
# 비영업승하차  -336.5467    518.209     -0.649      0.517   -1358.432     685.338
# 환승노선수      -0.1107      0.037     -2.989      0.003      -0.184      -0.038
# 스크린수        -0.3333      0.041     -8.112      0.000      -0.414      -0.252
# 포스터수        -0.1799      0.038     -4.731      0.000      -0.255      -0.105
# 대합실면적      -0.0096      0.038     -0.251      0.802      -0.085       0.066
# 정거장깊이      -0.0423      0.038     -1.114      0.267      -0.117       0.033
# 대학여부         0.0273      0.080      0.341      0.733      -0.131       0.185
# ==============================================================================
# Omnibus:                        6.529   Durbin-Watson:                   2.245
# Prob(Omnibus):                  0.038   Jarque-Bera (JB):                6.736
# Skew:                          -0.315   Prob(JB):                       0.0345
# Kurtosis:                       3.610   Cond. No.                     1.50e+05
# ==============================================================================


# =============================================================================
# 7. 앙상블 모형
# =============================================================================
# 7-1) train/test 분리 
en_train_x, en_test_x, en_train_y, en_test_y = train_test_split(station_com_drop, station_y, random_state=0)

# 7-2) rf_r
from sklearn.ensemble import RandomForestRegressor as rf_r
m_rfr = rf_r()
m_rfr.fit(en_train_x, en_train_y)
m_rfr.score(en_train_x, en_train_y)     # 94.84
m_rfr.score(en_test_x, en_test_y)       # 65.63

# 7-3) gb_r
from sklearn.ensemble import GradientBoostingRegressor as gb_r
m_gbr = gb_r()
m_gbr.fit(en_train_x, en_train_y)
m_gbr.score(en_train_x, en_train_y)     # 98.56
m_gbr.score(en_test_x, en_test_y)       # 64.25


# =============================================================================
# 8 잔차 정규성 검정
# =============================================================================
# 8-1) 잔차 히스토그램
plt.figure()
plt.hist(m_ols.resid)

# 8-2) 잔차 히스토그램 + kde 확인
import seaborn as sb
sb.distplot(m_ols.resid)    

# 8-3) qqplot
# 정규분포를 따른다면 점들이 빨간선 근처에 위치해야 한다.
from scipy import stats
stats.probplot(m_ols.resid, plot = plt)

# 8-3) 샤피로-윌크 검정(정규성 검정)
# H0 : 정규분포를 따른다
# H1 : 정규분포를 따르지 않는다
from scipy import stats
stats.shapiro(m_ols.resid)
# 결과값 : ShapiroResult(statistic=0.9741702079772949, pvalue=0.0006648256676271558)
# 귀무가설 채택! 따라서 정규분포를 따른다

# =============================================================================
# 9. 상관계수 시각화
# =============================================================================
# 9-1) 환경설정
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus= False)

# 9-2) heatmap
import seaborn as sb
sb.heatmap(np.abs(df_station.corr()),           # 데이터 2차원
           annot = True,                        # 주석(상관계수) 출력 여부
           cmap = 'Reds')                       # 컬러맵
# 급지분류와 상관계수 0.35 이상 목록
# 일평균승하차, 영업일평균승하차, 비영업일평균승하차, 스크린수, 포스터수

# 2) boxplot
plt.figure()
plt.boxplot(station_com_drop)

# 3) 전체적 분포 확인
station_com_drop.hist()    # 전체적 분포가 왼쪽에 치우침
pd.plotting.scatter_matrix(station_com_drop, s = 25)


# =============================================================================
# 10. 로지스틱 회귀분석(logistic regressor)
# =============================================================================
station_com
station_y
# 0. 범주형 컬럼 설정
station_com['환승노선수'] = station_com['환승노선수'].astype('category')
station_com['형식'] = station_com['형식'].astype('category')
station_com['층수'] = station_com['층수'].astype('category')
station_com['대학여부'] = station_com['대학여부'].astype('category')
station_y = station_y.astype('category')
station_com.info()

# 1. 이진분류를 위한 데이터 변환
station_y = Series(station_y).replace(2,1)         # 1,2급지 : 1 / 나머지 : 3

# 1. train/test 데이터 분리
from sklearn.model_selection import train_test_split
st_train_x, st_test_x, st_train_y, st_test_y = train_test_split(station_com, station_y, random_state=0)

# 2. 모델링
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st_train_x, st_train_y)
m_lr.score(st_train_x, st_train_y)          # 87.26

# 2-1) 계수 확인
m_lr.coef_
# [[-0.57252538, -0.21615675, -0.50023027, -0.75379686, -1.130624,
#   -1.12824522,  0.14127167, -0.55552547, -0.00649696, -0.58728085,
#   -0.10999151, -0.07880981, -0.08312097]]
m_lr.intercept_     # 1.30830561
m_lr.decision_function(st_train_x.iloc[:5, :])
# 2-2)회귀 식
# 1.30830561 - 0.57252538*X1 -0.21615675*X2 - 0.50023027*X3 - 0.75379686*X4 ...

# 3. 예측
new_data = station_com.iloc[0:1,:]
m_lr.predict(new_data)            # cut off에 해석된 Y 예상값
m_lr.predict_proba(new_data)      # 실제 확률 / 1or2급지 : 0.04463863, 3급지 : 0.95536137 

# 4. 시각화
p1 = m_lr.predict_proba(station_com)[:,1] 
p1.sort()
plt.plot(p1)

# 5. cut off 변화에 따른 예측률
m_lr.predict(new_data)        # cut off가 0.5로 설정되어 있음
p1 = m_lr.predict_proba(station_com)[:,1] 

cut_off = 0.5
predict_value = np.where(p1 > cut_off, 3, 1)
predict_value == m_lr.predict(station_com) 
station_y[predict_value == station_y].count()    # 180/210    # 85.71


# =============================================================================
# Restart
# =============================================================================
# 0. 불필요 컬럼 제거
station_x = station.drop(['호선', '역명', '급지분류', '광역시', '행정동', '월평균_승하차',
                          '영업일_평균_승하차', '승강장면적(m^2)', '형식', '층수'], axis=1)
# 0-1) 컬럼명 변경
station_x.columns = Series(station_x.columns).map(lambda x : x.split('(')[0]).map(lambda x : x.replace('_', ''))


# =============================================================================
# 1. 더미변수 분리 및 범주형 데이터 변환
# =============================================================================
len(station_x.columns)  # 10
station_x_sc_bf = station_x.iloc[:, [0,1,2,4,5,6,7]]
station_x_dummies = station_x.iloc[:, [3,8,9]]
station_x_dummies['환승노선수'] = station_x_dummies['환승노선수'].astype('category')
station_x_dummies['대학여부'] = station_x_dummies['대학여부'].astype('category')
station_x_dummies['상가여부'] = station_x_dummies['상가여부'].astype('category')

# =============================================================================
# 2. 더미변수 제외 변수 스케일링(StandardScaler)
# =============================================================================
# 2-1) 표준화
from sklearn.preprocessing import StandardScaler as standard
m_sc = standard()
station_x_sc_af = m_sc.fit_transform(station_x_sc_bf)
station_x_sc = DataFrame(data = station_x_sc_af, columns = station_x_sc_bf.columns)

# 2-2) 표준화 컬럼 + 더미변수 컬럼 합치기
station_com = pd.concat([station_x_sc, station_x_dummies], axis = 1)
station_y = station['급지분류'].astype('category')

# =============================================================================
# 3. Decision Tree
# =============================================================================
from sklearn.tree import DecisionTreeClassifier as dt_c
from sklearn.model_selection import train_test_split
# 3-1) train / test 분리
st_train_x, st_test_x, st_train_y, st_test_y = train_test_split(station_com, station_y, random_state=0)

# 3-2) model 생성 및 학습
m_dt = dt_c()
m_dt.fit(st_train_x, st_train_y)

# 3-3) score 확인
m_dt.score(st_train_x, st_train_y)          # 1.0   << 과적합
m_dt.score(st_test_x, st_test_y)            # 73.58

# 3-4) 변수 중요도 확인
dt_fe_im = m_dt.feature_importances_
# array([0.38622525, 0.099658  , 0.05299961, 0.10078057, 0.01135706,
#        0.2378346 , 0.02749578, 0.04788894, 0.        , 0.01892843,
#        0.        , 0.        , 0.01683175])
DataFrame(data = dt_fe_im.reshape(-1,1), index = station_com.columns)
# 주민등록인구수    0.080205
# 일평균승하차     0.304618
# 비영업일평균승하차  0.137299
# 스크린수       0.234671
# 포스터수       0.064348
# 대합실면적      0.039263
# 정거장깊이      0.092218
# 환승노선수      0.027496
# 대학여부       0.000000
# 상가여부       0.019883

# 3-5) 변수 컬럼 튜닝
# 비영업일평균승하차 => 불필요
# 대합실면적, 환승노선수, 대학여부, 상가여부 => 불필요
# 스크린수 & 포스터수 컬럼 => 결합 진행
station_x_sc_bf = station_x.iloc[:, [0,1,4,5,7]]
station_x_sc_bf['스크린포스터'] = list(map(lambda x, y : x+y, station_x_sc_bf['스크린수'], station_x_sc_bf['포스터수']))
station_com_ad1 = station_x_sc_bf.iloc[:, [0,1,5,4]]

# 3-6) re scaling
m_sc = standard()
station_com_ad1_sc = m_sc.fit_transform(station_com_ad1)

# 3-7) re modeling
st_sc_tr_x, st_sc_te_x, st_sc_tr_y, st_sc_te_y = train_test_split(station_com_ad1_sc, station_y, random_state=0)
m_dt = dt_c()
m_dt.fit(st_sc_tr_x, st_sc_tr_y)
m_dt.score(st_sc_tr_x, st_sc_tr_y)          # 1.0   << 과적합
m_dt.score(st_sc_te_x, st_sc_te_y)          # 69.81

# 3-8) 변수 중요도 확인
dt_fe_im = m_dt.feature_importances_
DataFrame(data = dt_fe_im.reshape(-1,1), index = station_com_ad1.columns)
# 주민등록인구수  0.069611
# 일평균승하차    0.510577
# 스크린포스터    0.263092
# 정거장깊이      0.156720

# =============================================================================
# 4. 로지스틱 회귀분석(logistic regressor)
# =============================================================================
station_com_ad1_sc
station_y
# 1. 이진분류를 위한 데이터 변환
station_y = Series(station_y).replace(2,1)         # 1,2급지 : 1 / 나머지 : 3

# 2. train/test 데이터 분리
from sklearn.model_selection import train_test_split
st_sc_tr_x, st_sc_te_x, st_sc_tr_y, st_sc_te_y = train_test_split(station_com_ad1_sc, station_y, random_state=0)

# 2. 모델링
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st_sc_tr_x, st_sc_tr_y)
m_lr.score(st_sc_tr_x, st_sc_tr_y)          # 87.89

# 2-1) 계수 확인
m_lr.coef_           # 0.04539174, -1.83745971, -1.42144103,  0.07530981
m_lr.intercept_      # 0.29480406
m_lr.decision_function(st_sc_tr_x[:5, :])
# 2-2)회귀 식
# 0.29480406 + 0.04539174*X1 -1.83745971*X2 - 1.42144103*X3 - 0.07530981*X4

# 3. 예측
new_data = st_sc_tr_x[0:1,:]
m_lr.predict(new_data)            # 3
m_lr.predict_proba(new_data)      # 확률 / 1or2급지 : 0.0404554, 3급지 : 0.9595446

# 4. 시각화
p1 = m_lr.predict_proba(station_com_ad1_sc)[:,1] 
p1.sort()
plt.plot(p1)

# 5. cut off 변화에 따른 예측률
m_lr.predict(new_data)        # cut off가 0.5로 설정되어 있음
p1 = m_lr.predict_proba(station_com_ad1_sc)[:,1] 

cut_off = 0.5
predict_value = np.where(p1 > cut_off, 3, 1)
predict_value == m_lr.predict(station_com_ad1_sc) 
station_y[predict_value == station_y].count()    # 181/210    # 86.19


# =============================================================================
# 5. 다중분류 로지스틱 회귀분석
# =============================================================================
station_com_ad1_sc
station_y
# 5-1) train/test 분리
from sklearn.model_selection import train_test_split
st_sc_tr_x, st_sc_te_x, st_sc_tr_y, st_sc_te_y = train_test_split(station_com_ad1_sc, station_y, random_state=0)

# 5-2) 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr_1 = lr()
m_lr_1.fit(st_sc_tr_x, st_sc_tr_y)
m_lr_1.score(st_sc_tr_x, st_sc_tr_y)    # 80.89

# 5-3) test 데이터 예측값 확인
m_lr_1.predict(st_sc_te_x[:5])          # [1, 3, 3, 2, 2]
m_lr_1.predict_proba(st_sc_te_x[:5]).round(4)
# [0.9913, 0.0087, 0.    ],
# [0.0236, 0.2938, 0.6825],
# [0.0024, 0.071 , 0.9266],
# [0.3307, 0.6196, 0.0498],
# [0.2348, 0.5487, 0.2165]

# 5-4) test 데이터 실제값 확인
st_sc_te_y[:5]                          # [1,3,3,2,1]

# 5-5) softmax 함수 활용 z값을 확률로 변환
decision = m_lr_1.decision_function(st_sc_te_x[:5])
np.round(decision, decimals=2)
# array([[ 5.81,  1.08, -6.9 ],
#        [-1.96,  0.56,  1.4 ],
#        [-3.11,  0.27,  2.84],
#        [ 0.42,  1.05, -1.47],
#        [-0.26,  0.59, -0.34]])

from scipy.special import softmax
proba = softmax(decision, axis=1)
np.round(proba, decimals=3)
# array([[0.991, 0.009, 0.   ],
#        [0.024, 0.294, 0.683],
#        [0.002, 0.071, 0.927],
#        [0.331, 0.62 , 0.05 ],
#        [0.235, 0.549, 0.217]])

# 5-3 결과와 5-5 결과가 동일함을 확인
# np.round(proba, decimals=3) 와 m_lr_1.predict_proba(st_sc_te_x[:5]).round(4) 일치함


# =============================================================================
# 6. 교차 검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(station_com_ad1_sc):
    train_x, test_x = station_com_ad1_sc[train_index], station_com_ad1_sc[test_index]
    train_y, test_y = station_y[train_index], station_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 75.72
np.mean(test_score1)         # 74.81

tr_score = []; te_score = []
# 교차검증을 통해 얻은 평가점수를 기반으로 가장 좋은 k는?
for train_index, test_index in m_cv1.split(station_com_ad1_sc):
    # k번 train / test 데이터 셋 분리
    train_x, test_x = station_com_ad1_sc[train_index], station_com_ad1_sc[test_index]
    train_y, test_y = station_y[train_index], station_y[test_index]
    
    # 선택된 데이터에 대해 C의 변화 확인
    train_score1 = []; test_score1 = []
    for i in [0.001, 0.01, 0.1, 1, 10, 100, 1000] :
        m_lr = lr(C = i)
        m_lr.fit(train_x, train_y)
        
        train_score1.append(m_lr.score(train_x, train_y))
        test_score1.append(m_lr.score(test_x, test_y))
    
    tr_score.append(train_score1)
    te_score.append(test_score1)

np.array(tr_score).shape
np.array(te_score).shape

train_sc = np.array(tr_score).mean(axis=0)
test_sc = np.array(te_score).mean(axis=0)

import matplotlib.pyplot as plt
plt.plot([0.001, 0.01, 0.1, 1, 10, 100, 1000], train_sc, c= 'r', label='Train_Score')
plt.plot([0.001, 0.01, 0.1, 1, 10, 100, 1000], test_sc, c= 'b', label='Test_Score')
plt.legend()
plt.title('교차검증 후 C의 변화에 따른 점수변화')   # 없음을 확인

# =============================================================================
# 6호선 데이터까지 합쳐서 스케일링 진행 후 로지스틱 회귀분석 진행
# =============================================================================
# 주민등록인구수 일평균승하차 스크린포스터 정거장깊이
no_6 = pd.read_csv('6호선제외_컬럼작업중_230323.csv')
in_6 = pd.read_csv('6호선_컬럼작업중_230323.csv')

# 1) 불필요 컬럼 삭제
st_no_6 = no_6.drop(['호선', '역명', '급지분류', '광역시','행정동', '월평균_승하차',
                     '영업일_평균_승하차', '비영업일_평균_승하차', '환승노선수', '대합실면적(m^2)',
                     '승강장면적(m^2)', '형식', '층수', '대학여부', '상가여부'], axis=1)
st_in_6 = in_6.drop(['역명', '급지분류', '광역시','행정동', '월평균_승하차',
                     '영업일_평균_승하차', '비영업일_평균_승하차', '환승노선수', '대합실면적(m^2)',
                     '승강장면적(m^2)', '형식', '층수', '대학여부', '상가여부'], axis=1)

# 2) 컬럼명 수정
st_no_6 = st_no_6.rename(columns={'정거장깊이(m)':'정거장깊이'})
st_in_6 = st_in_6.rename(columns={'정거장깊이(m)':'정거장깊이'})

# 3) 스크린 포스터 합치기 + 개별 컬럼 삭제
st_no_6['스크린포스터'] = list(map(lambda x, y : x + y, st_no_6['스크린수'], st_no_6['포스터수']))
st_in_6['스크린포스터'] = list(map(lambda x, y : x + y, st_in_6['스크린수'], st_in_6['포스터수']))
st_no_6 = st_no_6.drop(['스크린수', '포스터수'], axis=1)
st_in_6 = st_in_6.drop(['스크린수', '포스터수'], axis=1)

# 4) DF 합치기
station_sc_bf = pd.concat([st_no_6, st_in_6])

# 5) 스케일링
from sklearn.preprocessing import StandardScaler as standard
m_sc = standard()
station_sc_af = m_sc.fit_transform(station_sc_bf)
station_x_sc = DataFrame(data = station_sc_af, columns = station_sc_bf.columns)

# 6) 6호선 데이터만 별도 분리
station_x_sc_no6 = station_x_sc.iloc[:210, :]   # 6호선 제외
station_x_sc_in6 = station_x_sc.iloc[210:, :]
station_x_sc_in6.reset_index(drop=True, inplace=True)
station_x_sc_in6                                # 6호선

# 7) 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr_1 = lr()
m_lr_1.fit(station_x_sc_no6, station_y)
m_lr_1.score(station_x_sc_no6, station_y)    # 75.23

# 8) 튜닝
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 로지스틱 회귀 모델을 정의합니다.
logreg = LogisticRegression()

# 규제 하이퍼파라미터인 C와 penalty의 후보값들을 지정합니다.
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}

# GridSearchCV 객체를 생성합니다.
grid_search = GridSearchCV(logreg, param_grid, cv=5)

# 그리드 서치를 수행합니다.
grid_search.fit(station_x_sc_no6, station_y)

# 최적의 하이퍼파라미터와 정확도를 출력합니다.
print("최적의 하이퍼파라미터: {}".format(grid_search.best_params_))
# 최적의 하이퍼파라미터: {'C': 1, 'penalty': 'l2'}
print("최고 교차 검증 점수: {:.2f}".format(grid_search.best_score_))
# 최고 교차 검증 점수: 0.75

# 9) test(6호선) 데이터 예측값 확인
m_lr_1.predict(station_x_sc_in6)
m_lr_1.predict_proba(station_x_sc_in6)
# [0.07379382, 0.22547144, 0.70073474],       3
# [0.13802986, 0.40274661, 0.45922353],       3
# [0.05300647, 0.23113636, 0.71585717],       3
# [0.10233293, 0.28732302, 0.61034404],       3
# [0.02767394, 0.13314475, 0.8391813 ],       3
# [0.19941589, 0.4148803 , 0.38570382],       2
# [0.00525493, 0.03839353, 0.95635154],       3
# [0.04968298, 0.23224842, 0.7180686 ],       3
# [0.23226214, 0.43404736, 0.3336905 ],       2
# [0.13874602, 0.42569081, 0.43556316],       3
# [0.17074424, 0.37315933, 0.45609643],       3
# [0.11547043, 0.36843575, 0.51609381],       3
# [0.05320584, 0.13939057, 0.80740359],       3
# [0.0863853 , 0.31903046, 0.59458424],       3
# [0.02306223, 0.15530582, 0.82163195],       3
# [0.13558814, 0.31025797, 0.55415389],       3
# [0.03733807, 0.18842407, 0.77423786],       3
# [0.04944589, 0.21638101, 0.73417311],       3
# [0.08952637, 0.27328734, 0.63718629],       3
# [0.06897916, 0.28282748, 0.64819336],       3
# [0.04586652, 0.19232784, 0.76180564],       3
# [0.02644291, 0.14416684, 0.82939025],       3
# [0.0172486 , 0.13042888, 0.85232252],       3
# [0.0033096 , 0.037578  , 0.9591124 ],       3
# [0.00244637, 0.03133667, 0.96621696],       3
# [0.013798  , 0.09516325, 0.89103875],       3
# [0.05762493, 0.23012939, 0.71224567],       3
# [0.12913761, 0.36752562, 0.50333677],       3
# [0.03200804, 0.14701207, 0.8209799 ],       3
# [0.0251707 , 0.1707672 , 0.8040621 ],       3
# [0.02132322, 0.12501374, 0.85366305],       3
# [0.00280179, 0.04189713, 0.95530107],       3
# [0.02479933, 0.13739508, 0.83780558],       3
# [0.0405641 , 0.18572253, 0.77371337],       3
# [0.02127393, 0.14147115, 0.83725492],       3
# [0.06391513, 0.23708387, 0.699001  ],       3
# [0.01244705, 0.10554895, 0.882004  ]        3

line6_result = in_6.iloc[:, 0:2]
line6_result['급지분류'] = m_lr_1.predict(station_x_sc_in6)
# 해당 급지의 확률 출력 함수
def proba_f(x, y):
    return m_lr_1.predict_proba(station_x_sc_in6)[x, y-1] * 100
line6_result['급지확률'] = list(map(proba_f, Series(line6_result.index), line6_result['급지분류']))
line6_result


station_x_sc_no6.to_csv('6호선제외_scaling_230327.csv')
station_x_sc_in6.to_csv('6호선_scaling_230327.csv')
line6_result.to_csv('6호선_result_230327.csv')



