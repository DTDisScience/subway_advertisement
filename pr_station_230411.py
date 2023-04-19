# -*- coding: utf-8 -*-
%run my_profile.py

# =============================================================================
# 데이터불러오기 및 데이터프레임 작업
# =============================================================================
# =============================================================================
# 독립변수 작업
# =============================================================================
df1 = pd.read_csv('subway1_8.csv')

# 3호선 충무로 : 일평균_승하차 0으로 제외
df1 = df1.drop(77)
df1.reset_index(drop=True, inplace=True)

# 등급분류 없는 데이터 8개 삭제
# 학여울(3호선), 동역공(4), 개화산(5), 마장(5), 마천(5), 방화(5), 오금(5), 장암(7)
df1.loc[pd.isnull(df1['등급']) == True, ['호선', '역명']]
df1 = df1.drop([77,81,101,116,117,123,137,182])
df1.reset_index(drop=True, inplace=True)

# 불필요 컬럼 제거
df2 = df1.loc[:, ['주민등록인구수', '일평균_승하차', '승강장면적(m^2)', '대합실면적(m^2)']]

# 컬럼명 변환(특수기호, 괄호 안 내용 제거)
df2.columns = ['주민등록인구수', '일평균승하차', '승강장면적', '대합실면적']

# 승강장면적, 대합실면적 round 적용 (소수점값 버림)
df2['승강장면적'] = df2['승강장면적'].map(lambda x : round(x))
df2['대합실면적'] = df2['대합실면적'].map(lambda x : round(x))

# 밀도 컬럼 추가 (밀도 = 일평균승하차 / 승강장면적)
df2['밀도'] = list(map(f1, df2['일평균승하차'], df2['승강장면적']))
def f1(x, y) :
    return round(x/y)

# 승강장면적 컬럼 제거
df2 = df2.drop('승강장면적', axis=1)

# =============================================================================
# 종속변수 2진분류
# =============================================================================
df1['등급'] = df1['등급'].map(lambda x : 1 if x == 2 else x).map(lambda x : int(2) if x != 1 else int(x))
df_y = df1['등급'].astype('category')


# =============================================================================
# 변수 로그변환
# =============================================================================
# 히스토그램으로 변환 전후 비교
df2.hist()
np.log1p(df2).hist()

df_x = np.log1p(df2)


# =============================================================================
# train/test 분리
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_x.values, df_y, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7466666666666667
m_lr.score(test_x, test_y)             # 0.803921568627451


# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df_x.values):
    train_x, test_x = df_x.values[train_index], df_x.values[test_index]
    train_y, test_y = df_y[train_index], df_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7512030905077263
np.mean(test_score1)         # 0.7159803921568628








