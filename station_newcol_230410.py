# -*- coding: utf-8 -*-


station_pcd = pd.read_clipboard()
# 274

station_pcd['2020^1101053010001^to_fa_010^336']
station_pcd.columns = ['작업전']
station_pcd['코드'] = station_pcd['작업전'].map(lambda x : x.split('^')[1])
station_pcd['수'] = station_pcd['작업전'].map(lambda x : int(x.split('^')[3]))
station_pcd = station_pcd.drop('작업전', axis=1)
station_pcd['행정구역'] = station_pcd['코드'].map(lambda x : x[:8])

station_pcd.groupby('행정구역')[['수']].sum()


df1

# =============================================================================
# 230411 (화)
# =============================================================================
df1 = pd.read_csv('subway1_8.csv')
df1.to_csv('6호선제외_컬럼작업중_230323.csv', encoding = 'cp949')

df1 = df1.drop(77)
df1 = df1.drop([77, 81,101,116,117,123,137,182])
df1.reset_index(drop=True, inplace=True)
df2 = df1.loc[:, ['호선', '역명', '행정동', '주민등록인구수', '일평균_승하차', '승강장면적(m^2)',
                  '등급']]

# =============================================================================
# df2 : 행정동 변경 및 역 영향권에 따른 주민등록인구수 수정 필요
# =============================================================================
df2['등급'] = df2['등급'].map(lambda x : int(x)).astype('category')
df3 = df2.drop(['호선', '역명', '행정동', '등급'], axis=1)
df3.columns = ['주민등록인구수', '일평균승하차', '승장장면적']
df3['승장장면적'] = df3['승장장면적'].map(lambda x : round(x))
df3

from sklearn.preprocessing import MinMaxScaler as minmax
m_mm = minmax()
x_sc = m_mm.fit_transform(df3)
x_sc
plt.figure()
plt.boxplot(x_sc)
DataFrame(x_sc).hist()       # 매우 심한 좌측 편포

DataFrame(np.log1p(x_sc)).hist()

np.log1p(df3).hist()

df4 = np.log1p(df3)
df4

# =============================================================================
# y 등급 2진 분류로 변환
# =============================================================================
df_y = df2['등급'].map(lambda x : 1 if x == 2 else x).map(lambda x : 2 if x != 1 else x)
df_y = df_y.astype('category')
df_y


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df4.values, df_y, random_state=0)

from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.6733333333333333
m_lr.score(test_x, test_y)             # 0.7647058823529411


# =============================================================================
# 4분위수 변환
# =============================================================================
DataFrame(x_sc).describe()





# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df4.values):
    train_x, test_x = df4.values[train_index], df4.values[test_index]
    train_y, test_y = df_y[train_index], df_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 70.80
np.mean(test_score1)         # 66.10



# =============================================================================
# 승하차인원/승강장면적 : 인구밀도; 대합실면적 사용
# =============================================================================
df5 = df1.loc[:, ['주민등록인구수', '일평균_승하차', '승강장면적(m^2)',
                  '대합실면적(m^2)', '등급']]

df5.columns = ['주민등록인구수', '일평균승하차', '승강장면적', '대합실면적', '등급']
df5['밀도'] = list(map(f1, df5['일평균승하차'], df5['승강장면적']))

def f1(x, y) :
    return round(x/y)

df6 = df5.drop('승강장면적', axis=1)
df6['대합실면적'] = df6['대합실면적'].map(lambda x : int(x))
df6_x = df6.iloc[:, [0,1,2,4]]
df6_x_1 = df6.iloc[:, [0,1,2]]
df6_y = df6['등급'].map(lambda x : 1 if x == 2 else x).map(lambda x : 2 if x != 1 else x).astype('category')

from sklearn.preprocessing import MinMaxScaler as minmax
m_mm = minmax()
df6_x_sc = m_mm.fit_transform(df6_x)
df6_x_sc
plt.figure()
plt.boxplot(df6_x_sc)
DataFrame(df6_x_sc).hist()       # 매우 심한 좌측 편포

DataFrame(np.log1p(df6_x_sc)).hist()

np.log1p(df6_x).hist()

df7 = np.log1p(df6_x)
df7
df7.hist()
# =============================================================================
# 
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df7.values, df6_y, random_state=0)

from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7549668874172185
m_lr.score(test_x, test_y)             # 0.7


# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df7.values):
    train_x, test_x = df7.values[train_index], df7.values[test_index]
    train_y, test_y = df6_y[train_index], df6_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 75.12
np.mean(test_score1)         # 71.59
m_dt = dt_c()
m_dt.fit(df7.values, df6_y)
m_dt.feature_importances_


# =============================================================================
# 3진분류
# =============================================================================
df7_y = df6['등급'].map(lambda x : 1 if x == 2 else x).map(lambda x : 2 if x == 3 else x).map(lambda x : int(3) if x == 4 else int(x)).astype('category')
df7_y = df1['급지분류'].astype('category')

# =============================================================================
# 
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df7.values, df7_y, random_state=0)

from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7549668874172185
m_lr.score(test_x, test_y)             # 0.7

from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df7.values):
    train_x, test_x = df7.values[train_index], df7.values[test_index]
    train_y, test_y = df7_y[train_index], df7_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 75.12
np.mean(test_score1)         # 71.59








