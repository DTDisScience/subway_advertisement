# -*- coding: utf-8 -*-
%run my_profile.py

df_original; df_original_log
df_predict = df_original.iloc[:, [0,1,12]]
df_original.to_csv('df_orginal_230418.csv', encoding='cp949')
df_ori_x = df_original.loc[:, ['출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원']]
df_log_x = df_original_log.loc[:, ['출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원']].values
df_log_y = df_original['등급'].values

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_log_x, df_log_y, random_state=0)

# =============================================================================
# 로지스틱 회귀모델
# '출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원'
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import KFold
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)
m_lr.score(test_x, test_y)

# 교차검증을 통해, 각 데이터가 한 번씩 테스트 데이터로 사용됨
# 각 데이터가 테스트 데이터로 사용되었을 때, 그 예측값 출력하기
# 또한, 각 데이터가 index 순서대로 test 데이터로 사용됨을 확인
# 다른 모델에서는 predict_values 만 출력
lr_predict_values = []; test_values = []; test_y_index = []
m_cv1 = KFold(4)
for train_index, test_index in m_cv1.split(df_log_x):
    train_x, test_x = df_log_x[train_index], df_log_x[test_index]
    train_y, test_y = df_log_y[train_index], df_log_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    lr_predict_values.append(m_lr.predict(test_x))
    test_values.append(test_y)
    test_y_index.append(list(test_index))

# 단일 리스트로 변환
lr_predict_values_list = []; test_values_list = []; test_y_index_list = []
for i in range(0, 4) :
    for j in range(0, len(lr_predict_values[i])):
        lr_predict_values_list.append(lr_predict_values[i][j])
        test_values_list.append(test_values[i][j])
        test_y_index_list.append(test_y_index[i][j])
    
df_predict['로지스틱_회귀모델_예측값'] = lr_predict_values_list


# =============================================================================
# RandomForestClassifier
# 트리수 45, max_depth=3 고정
# =============================================================================
from sklearn.ensemble import RandomForestClassifier as rf_c
rfc_predict_values = []
m_cv1 = KFold(4)
for train_index, test_index in m_cv1.split(df_log_x):
    train_x, test_x = df_log_x[train_index], df_log_x[test_index]
    train_y, test_y = df_log_y[train_index], df_log_y[test_index]

    m_rfc = rf_c(n_estimators = 45, max_depth=3)
    m_rfc.fit(train_x, train_y)
    
    rfc_predict_values.append(m_rfc.predict(test_x))

# 단일 리스트로 변환
rfc_predict_values_list = []
for i in range(0, 4) :
    for j in range(0, len(rfc_predict_values[i])):
        rfc_predict_values_list.append(rfc_predict_values[i][j])

df_predict['랜덤포레스트_예측값'] = rfc_predict_values_list


# =============================================================================
# GradientBoostingClassifier
# 트리수 25, max_depth=3 고정
# =============================================================================
from sklearn.ensemble import GradientBoostingClassifier as gb_c
m_gbc = gb_c(n_estimators=25, max_depth=3)

gbc_predict_values = []
m_cv1 = KFold(4)
for train_index, test_index in m_cv1.split(df_log_x):
    train_x, test_x = df_log_x[train_index], df_log_x[test_index]
    train_y, test_y = df_log_y[train_index], df_log_y[test_index]

    m_gbc = gb_c(n_estimators=25, max_depth=3)
    m_gbc.fit(train_x, train_y)
    
    gbc_predict_values.append(m_gbc.predict(test_x))

# 단일 리스트로 변환
gbc_predict_values_list = []
for i in range(0, 4) :
    for j in range(0, len(gbc_predict_values[i])):
        gbc_predict_values_list.append(gbc_predict_values[i][j])

df_predict['GradientBoost_predict'] = gbc_predict_values_list


# =============================================================================
# XGBClassifier
# 트리수 20, max_depth=3 고정
# =============================================================================
# ***** 주의!!! *****
# XGB의 경우, 종속변수의 0과 1 구분만 가능!
# 따라서 y : 기존2를 0으로 변경 및
# 예측값의 경우 0으로 예측한 값을 2로 변경!
df_log_y_xgb = Series(df_log_y).map(lambda x : 0 if x == 2 else x).values
from xgboost.sklearn import XGBClassifier
m_xgb = XGBClassifier(n_estimators=20, max_depth=3)

xgb_predict_values = []
m_cv1 = KFold(4)
for train_index, test_index in m_cv1.split(df_log_x):
    train_x, test_x = df_log_x[train_index], df_log_x[test_index]
    train_y, test_y = df_log_y_xgb[train_index], df_log_y_xgb[test_index]

    m_xgb = XGBClassifier(n_estimators=20, max_depth=3)
    m_xgb.fit(train_x, train_y)
    
    xgb_predict_values.append(m_xgb.predict(test_x))

# 단일 리스트로 변환
xgb_predict_values_list = []
for i in range(0, 4) :
    for j in range(0, len(xgb_predict_values[i])):
        # 예측값 0이면 2로 출력하도록 변경하기
        if xgb_predict_values[i][j] == 0 :
            xgb_predict_values_list.append(2)
        else :
            xgb_predict_values_list.append(1)
        
df_predict['XGBClassifier_predict'] = xgb_predict_values_list

df_predict.columns = ['호선', '역명', '실제등급', 'LogisticReg_predict',
                      'RFClassifier_predict', 'GradientBoost_predict', 'XGBClassifier_predict']

df_predict['number_of_correct'] = list(map(predict_count_sum, df_predict['실제등급'],
                                           df_predict['LogisticReg_predict'],
                                           df_predict['RFClassifier_predict'],
                                           df_predict['GradientBoost_predict'],
                                           df_predict['XGBClassifier_predict']))
def predict_count_sum(a, b, c, d, e):
    lr_score = abs(a-b)
    rfc_score = abs(a-c)
    gbc_score = abs(a-d)
    xgb_score = abs(a-e)
    
    return 4-(lr_score + rfc_score + gbc_score + xgb_score)

df_original.to_csv('df_orginal_230418.csv', encoding='cp949')
df_original = pd.read_cs('df_orginal_230418.csv', encoding='cp949')
df_predict.to_csv('predict_values_by_model_230418.csv', encoding='cp949')
df_predict = pd.read_csv('predict_values_by_model_230418.csv', encoding='cp949')
wrong_values_index = df_predict.loc[df_predict['number_of_correct'] == 0, :].index
df_fail = df_original.iloc[wrong_values_index, [0,1,12,2,4,5,7,8]]
df_success = df_original.drop(wrong_values_index).iloc[:, [0,1,12,2,4,5,7,8]]
df_fail.loc[df_fail['등급'] == 2, :]
# 0. 실제 2등급으로 분류되어 있으나, 1등급으로 예측한 데이터는 
#    일반적으로 대합실면적 유동인구에 비해 비교적 크다
# 1. 수인분당선, 경의선, 공항철도, 9호선 => 고려되지 않은 분류
# 2. 종점여부

df_fail.loc[df_fail['등급'] == 1, :]
# 주로 5, 7, 8호선 중 1등급으로 분류된 역을 예측하지 못함
# 1. 5호선 : 까치산, 김포공항, 발산, 화곡 ==> 강서구
#            주거지역이었으나, 마곡 업무지구의 등장(?)으로 해당 부분에 근거한 1등급 
#            분류로 예상됨.
#            특히 김포공항의 경우, 공항철도, 9호선에 집중되는 경향

# 2. 

df_success.loc[df_success['등급'] == 1, :]

from sklearn.cluster import KMeans
from sklearn.cluster import dbscan

# 3-1) KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(dff)
kmeans_cluster = Series(kmeans.labels_).map(lambda x : 2 if x == 0 else x)

Series(kmeans.labels_).value_counts()

df_predict = df_predict.drop('number_of_correct', axis=1)
df_predict['kmeans'] = kmeans_cluster
sum(list(map(lambda x, y : 0 if x == y else 1, df_predict['실제등급'], df_predict['kmeans'])))

# 3-2) dbscan
_, dbscan_cluster = dbscan(df_mm, eps = 0.4, min_samples = 30)
_, dbscan_cluster = dbscan(df_mm)
Series(dbscan_cluster).value_counts()
dbscan_predict = Series(dbscan_cluster).map(lambda x : 2 if x == 0 else 1)

heatmap(df_mm)
sb.heatmap(df_mm.corr(), annot = True, cmap = 'Reds')

df_mm.loc[:, ['출근시간대', '대합실면적', '사업체수', '종사자수']]
'퇴근시간대', '출근시간대승차인원', '출근시간대일반인하차', '사업체수', '종사자수'

_, dbscan_cluster = dbscan(df_mm.loc[:, ['출근시간대', '대합실면적', '사업체수']], eps = 0.4, min_samples = 30)
Series(dbscan_cluster).value_counts()
dbscan_predict = Series(dbscan_cluster).map(lambda x : 2 if x == 0 else 1)

dbscan_sc_1 = df_mm.loc[:, ['출근시간대', '대합실면적', '스타벅스수']]
dbscan_sc_2 = df_mm.loc[:, ['사업체수', '대합실면적', '스타벅스수']]
fig, ax = plt.subplots(1,2)
ax[1].scatter(dbscan_sc_1.iloc[:,0], dbscan_sc_1.iloc[:,1], dbscan_sc_1.iloc[:,2], c = dbscan_cluster)


import mglearn
from mpl_toolkits.mplot3d import Axes3D, axes3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(dbscan_sc_1.iloc[:,0],          # x축 좌표 (첫 번째 설명변수)
           dbscan_sc_1.iloc[:,1],          # y축 좌표 (두 번째 설명변수)
           dbscan_sc_1.iloc[:,2],          # z축 좌표 (세 번째 설명변수)
           c = dbscan_cluster,                 # 점 색
           cmap = mglearn.cm2,      # 투명도 (팔레트)
           s= 60,                   # size 
           edgecolors = 'k')        # 테두리 색

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(dbscan_sc_2.iloc[:,0],          # x축 좌표 (첫 번째 설명변수)
           dbscan_sc_2.iloc[:,1],          # y축 좌표 (두 번째 설명변수)
           dbscan_sc_2.iloc[:,2],          # z축 좌표 (세 번째 설명변수)
           c = dbscan_cluster,                 # 점 색
           cmap = mglearn.cm2,      # 투명도 (팔레트)
           s= 60,                   # size 
           edgecolors = 'k')        # 테두리 색


# =============================================================================
# 조합
# =============================================================================
from itertools import combinations
iterator = ['퇴근시간대', '출근시간대승차인원', '출근시간대일반인하차', '사업체수', '종사자수']
list(combinations(iterator, 3))[0].replace('')
df_mm.loc[:,list(combinations(iterator, 3))[0]]
len(list(combinations(iterator, 3)))
a1 = list(combinations(iterator, 3))

a1[1:5]
for i in range(0, len(a1)) :
    dbscan_sc = df_mm.loc[:, a1[i]]
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.scatter(dbscan_sc.iloc[:,0],          # x축 좌표 (첫 번째 설명변수)
                dbscan_sc.iloc[:,1],          # y축 좌표 (두 번째 설명변수)
                dbscan_sc.iloc[:,2],          # z축 좌표 (세 번째 설명변수)
                c = dbscan_cluster,                 # 점 색
                cmap = mglearn.cm2,      # 투명도 (팔레트)
                s= 60,                   # size 
                edgecolors = 'k') 


ax.set_title('DBSCAN')






