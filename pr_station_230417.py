# -*- coding: utf-8 -*-
%run my_profile.py
df1 = pd.read_csv('grid_total.csv', encoding='UTF-8', index_col=0)
df_x_230413 = pd.read_csv('승하차_일반인_컬럼작업완_230414.csv', encoding = 'cp949', index_col=0)
df1; df_x_230413

df_original = pd.concat([df1.iloc[:, :2], df_x_230413], axis=1)
df_original = pd.concat([df_original, df1['등급']], axis=1)
df_original_log = df_original[:]

fig, ax = plt.subplots(2,5)
for i in range(0, 10):
    if i < 5 :
        ax[0, i].scatter(df_original.iloc[:, i + 2], df_original['등급'], c = df_original['등급'])
        ax[0, i].set_title(df_original.columns[i + 2])
    else :
        ax[1, i-5].scatter(df_original.iloc[:, i + 2], df_original['등급'], c = df_original['등급'])
        ax[1, i-5].set_title(df_original.columns[i + 2])

for i in range(0, 10):
    df_original_log.iloc[:, i + 2] = df_original_log.iloc[:, i + 2].map(lambda x : np.log1p(x))

fig, ax = plt.subplots(2,5)
for i in range(0, 10):
    if i < 5 :
        ax[0, i].scatter(df_original_log.iloc[:, i + 2], df_original_log['등급'], c = df_original_log['등급'])
        ax[0, i].set_title(df_original_log.columns[i + 2])
    else :
        ax[1, i-5].scatter(df_original_log.iloc[:, i + 2], df_original_log['등급'], c = df_original_log['등급'])
        ax[1, i-5].set_title(df_original_log.columns[i + 2])


# =============================================================================
# 출근시간대, 대합실면적, 스타벅스수 3차원 산점도 & 초평면
# =============================================================================
import mglearn
from mpl_toolkits.mplot3d import Axes3D, axes3d
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df_original.loc[df_original['등급']==1, df_original.columns[2]],          # x축 좌표 (첫 번째 설명변수)
           df_original.loc[df_original['등급']==1, df_original.columns[4]],          # y축 좌표 (두 번째 설명변수)
           df_original.loc[df_original['등급']==1, df_original.columns[5]],          # z축 좌표 (세 번째 설명변수)
           c = 'b',                 # 점 색
           cmap = mglearn.cm2,      # 투명도 (팔레트)
           s= 60,                   # size 
           edgecolors = 'k')        # 테두리 색

ax.scatter(df_original.loc[df_original['등급']==2, df_original.columns[2]],          # x축 좌표 (첫 번째 설명변수)
           df_original.loc[df_original['등급']==2, df_original.columns[4]],          # y축 좌표 (두 번째 설명변수)
           df_original.loc[df_original['등급']==2, df_original.columns[5]],          # z축 좌표 (세 번째 설명변수)
           c = 'r',                 # 점 색
           cmap = mglearn.cm2,      # 투명도 (팔레트)
           s= 60,                   # size
           edgecolors = 'k')        # 테두리 색

df_svm = df_original.iloc[:, [2,4,5]].values
y = df_original['등급'].values


from sklearn.svm import SVC
m_svm = SVC(kernel='linear')
m_svm.fit(df_svm, y)

intercept = m_svm.intercept_      # 초평면 절편
coef = m_svm.coef_                # 초평면 기울기를 만들고 ravel() 로 평탄화

# 초평면 그리기
xx, yy = np.meshgrid(np.linspace(1, 60000, 30), np.linspace(1, 25000, 30))
zz = (-intercept[0]-coef[0][0]*xx-coef[0][1]*yy) / coef[0][2]
ax.plot_surface(xx, yy, zz, alpha=0.2)

# =============================================================================
# 로지스틱 회귀모델
# '출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원'
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()

from mlxtend.feature_selection import SequentialFeatureSelector
sfs = SequentialFeatureSelector(m_lr, k_features='best', forward=False, verbose=1, scoring='accuracy', cv=4)
sfs.fit(df_log_x, df_log_y)
a1 = list(sfs.k_feature_names_)
vcol = []
for i in a1 :
    vcol.append(int(i))
print("선택된 변수: ", cols[vcol])      # '출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원'
print("검증 정확도: ", sfs.k_score_)    # 0.7756862745098039

from sklearn.model_selection import KFold
train_score1 = []; test_score1 = []
for i in range(0, 100) :
    m_cv1 = KFold(4, shuffle=True, random_state=i)
    for train_index, test_index in m_cv1.split(df_log_x[:, vcol]):
        train_x, test_x = df_log_x[:, vcol][train_index], df_log_x[:, vcol][test_index]
        train_y, test_y = df_log_y[train_index], df_log_y[test_index]
    
        m_lr = lr()
        m_lr.fit(train_x, train_y)
        
        train_score1.append(m_lr.score(train_x, train_y))
        test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7797050772626932
np.mean(test_score1)         # 0.7667578431372548


# =============================================================================
# 독립변수 재정립
# =============================================================================
# ['출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원']
rfc_cols = cols[vcol]
gbc_cols = cols[vcol]
df_log_x_choosen = df_log_x[:, vcol]


# =============================================================================
# 랜덤포레스트
# =============================================================================
from sklearn.ensemble import RandomForestClassifier as rf_c
m_rfc = rf_c(n_estimators = 60, max_depth=3)

from sklearn.model_selection import KFold
# tree 수 변화에 따른 교차검증 점수 변화 확인
train_score = []; test_score = []
for i in range(10, 151, 5) :
    m_cv1 = KFold(4, shuffle=True)
    train_score1 = []; test_score1 = []
    for train_index, test_index in m_cv1.split(df_log_x_choosen):
        train_x, test_x = df_log_x_choosen[train_index], df_log_x_choosen[test_index]
        train_y, test_y = df_log_y[train_index], df_log_y[test_index]
    
        m_rfc = rf_c(n_estimators = i, max_depth=3)
        m_rfc.fit(train_x, train_y)
        
        train_score1.append(m_rfc.score(train_x, train_y))
        test_score1.append(m_rfc.score(test_x, test_y))
    train_score.append(np.mean(train_score1))
    test_score.append(np.mean(test_score1))

# 교차검증 후 최종 점수
plt.figure()
plt.plot(range(10,151,5), train_score, label='train_score')
plt.plot(range(10,151,5), test_score, label='test_score')
plt.title('RandomForestClassifier')

# 트리수 45, max_depth=3 고정
train_x, test_x, train_y, test_y = train_test_split(df_log_x_choosen, df_log_y, random_state=0)
m_rfc = rf_c(n_estimators = 45, max_depth=3)
m_rfc.fit(train_x, train_y)
m_rfc.score(train_x, train_y)    # 0.84
m_rfc.score(test_x, test_y)      # 0.7843137254901961
rfc_feature_importances = Series(m_rfc.feature_importances_)
rfc_cols
rfc_feature_importances_df = pd.concat([Series(rfc_cols), rfc_feature_importances], axis = 1)
rfc_feature_importances_df.columns = ['변수명', '변수중요도']
rfc_feature_importances_df = rfc_feature_importances_df.sort_values(['변수중요도'], ascending = False, ignore_index=True)
rfc_feature_importances_df
#          변수명      변수중요도
# 0      대합실면적     0.342430
# 1      스타벅스수     0.238874
# 2      출근시간대     0.162339
# 3  퇴근시간대승차인원  0.136743
# 4  출근시간대하차인원  0.119615


# =============================================================================
# GradientBoostingClassifier
# =============================================================================
from sklearn.ensemble import GradientBoostingClassifier as gb_c
m_gbc = gb_c()

# tree 수 변화에 따른 교차검증 점수 변화 확인
train_score = []; test_score = []
for i in range(10, 151, 5) :
    m_cv1 = KFold(4, shuffle=True)
    train_score1 = []; test_score1 = []
    for train_index, test_index in m_cv1.split(df_log_x_choosen):
        train_x, test_x = df_log_x_choosen[train_index], df_log_x_choosen[test_index]
        train_y, test_y = df_log_y[train_index], df_log_y[test_index]
    
        m_gbc = gb_c(n_estimators = i, max_depth=3)
        m_gbc.fit(train_x, train_y)
        
        train_score1.append(m_gbc.score(train_x, train_y))
        test_score1.append(m_gbc.score(test_x, test_y))
    train_score.append(np.mean(train_score1))
    test_score.append(np.mean(test_score1))
    
# max_depth=3 고정 / 트리수 변화에 따른 score 변화 확인
plt.figure()
plt.plot(range(10,151,5), train_score, label='train_score')
plt.plot(range(10,151,5), test_score, label='test_score')
plt.title('GradientBoostingClassifier')

# 트리수 25, max_depth=3 고정
train_x, test_x, train_y, test_y = train_test_split(df_log_x_choosen, df_log_y, random_state=0)
m_gbc = gb_c(n_estimators=25, max_depth=3)
m_gbc.fit(train_x, train_y)

m_gbc.score(train_x, train_y)    # 0.9333333333333333
m_gbc.score(test_x, test_y)      # 0.7450980392156863

gbc_feature_importances = Series(m_gbc.feature_importances_)
gbc_cols
gbc_feature_importances_df = pd.concat([Series(gbc_cols), gbc_feature_importances], axis = 1)
gbc_feature_importances_df.columns = ['변수명', '변수중요도']
gbc_feature_importances_df = gbc_feature_importances_df.sort_values(['변수중요도'], ascending = False, ignore_index=True)
gbc_feature_importances_df
#          변수명      변수중요도
# 0      스타벅스수     0.308694
# 1      대합실면적     0.276501
# 2      출근시간대     0.203592
# 3  퇴근시간대승차인원  0.108415
# 4  출근시간대하차인원  0.102799

# =============================================================================
# XGBClassifier
# =============================================================================
from xgboost.sklearn import XGBClassifier
from xgboost import XGBClassifier as XGB_C
from sklearn.model_selection import cross_val_score as cv
from xgboost import plot_importance

m_xgb = XGBClassifier()
xgb_y = df_log_y.reshape()
cv_score_xgb = cv(m_xgb, df_log_x_choosen, xgb_y, cv = 4, )
cv_score_xgb.mean()            # 7610784313725489
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score_xgb.mean(), cv_score_xgb.std() * 2))
# 0.76 (+/- 0.06)     교차검증 점수 평균 / 표준편차

# tree 수 변화에 따른 교차검증 점수 변화 확인
train_score = []; test_score = []
for i in range(10, 151, 5) :
    m_cv1 = KFold(4, shuffle=True)
    train_score1 = []; test_score1 = []
    for train_index, test_index in m_cv1.split(df_log_x_choosen):
        train_x, test_x = df_log_x_choosen[train_index], df_log_x_choosen[test_index]
        train_y, test_y = xgb_y[train_index], xgb_y[test_index]
    
        m_xgb = XGBClassifier(n_estimators = i, max_depth=3)
        m_xgb.fit(train_x, train_y)
        
        train_score1.append(m_xgb.score(train_x, train_y))
        test_score1.append(m_xgb.score(test_x, test_y))
    train_score.append(np.mean(train_score1))
    test_score.append(np.mean(test_score1))

# max_depth=3 고정 / 트리수 변화에 따른 score 변화 확인
plt.figure()
plt.plot(range(10,151,5), train_score, label='train_score')
plt.plot(range(10,151,5), test_score, label='test_score')
plt.title('XGBClassifier')

# 트리수 20, max_depth=3 고정
train_x, test_x, train_y, test_y = train_test_split(df_log_x_choosen, xgb_y, random_state=0)
m_xgb = XGBClassifier(n_estimators=20, max_depth=3)
m_xgb.fit(train_x, train_y)
m_xgb.score(train_x, train_y)  # 0.9466666666666667
m_xgb.score(test_x, test_y)    # 0.8235294117647058

# 교차검증 진행
m_cv1 = KFold(4)
train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df_log_x_choosen):
    train_x, test_x = df_log_x_choosen[train_index], df_log_x_choosen[test_index]
    train_y, test_y = xgb_y[train_index], xgb_y[test_index]

    m_xgb = XGBClassifier(n_estimators=20, max_depth=3)
    m_xgb.fit(train_x, train_y)
    
    train_score1.append(m_xgb.score(train_x, train_y))
    test_score1.append(m_xgb.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.938653421633554
np.mean(test_score1)         # 0.7607843137254902

m_xgb.feature_importances_
# ['출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원']
xgb_feature_importances = Series(m_xgb.feature_importances_)
xgb_cols = ['출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원']
xbc_feature_importances_df = pd.concat([Series(xgb_cols), xgb_feature_importances], axis = 1)
xbc_feature_importances_df.columns = ['변수명', '변수중요도']
xbc_feature_importances_df = xbc_feature_importances_df.sort_values(['변수중요도'], ascending = False, ignore_index=True)
xbc_feature_importances_df
#          변수명      변수중요도
# 0      스타벅스수     0.387518
# 1      대합실면적     0.188906
# 2      출근시간대     0.178866
# 3  퇴근시간대승차인원  0.132057
# 4  출근시간대하차인원  0.112653

plot_importance(m_xgb)

