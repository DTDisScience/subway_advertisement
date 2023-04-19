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


# =============================================================================
# 2023/04/12(수) 추가  - 인구수 데이터 update
# =============================================================================
# grid 기반 인구데이터 불러오기
pop_df = pd.read_csv('grid_03.csv', encoding='UTF-8')

# 이유는 모르겠으나 각 역의 환승노선수 만큼 중복 데이터가 존재함
# 다사5754,"92786","4712",보문,UI,"952","2639","504065","1134217",37.585286000000004,127.019380999999996
# 다사5754,"92786","2639",보문,"6","638","2639","504065","1134217",37.585286000000004,127.019380999999996
# 다사5754,"92786","4712",보문,UI,"952","2639","504065","1134217",37.585286000000004,127.019380999999996
# 다사5754,"92786","2639",보문,"6","638","2639","504065","1134217",37.585286000000004,127.019380999999996
# => 6 : 6호선 / UI : 우이신설선  ==> 각 2개씩 총 4개
# 따라서 중복 데이터 제거 필요

# 필요한 컬럼만 선택
pop_df_1 = pop_df.loc[:, ['호선', '전철역명', '2020년_인구_다사_1K_work_인구수']]

# 중복데이터 제거 / 기존 892개 행에서 638행으로 변환
pop_df_1 = pop_df_1.drop_duplicates(ignore_index = True)

# 조건설정을 통한 1~8호선 데이터만 선택
con1 = ['1', '2', '3', '4', '5', '6', '7', '8']
pop_df_1['호선'] = pop_df_1['호선'].map(lambda x : int(x) if x in con1 else 9)
pop_df_2 = pop_df_1.loc[pop_df_1['호선'] != 9, :]

# 호선, 전철역명 기준 오름차순 정렬
pop_df_3 = pop_df_2.sort_values(['호선', '전철역명'], ascending = [True, True])
pop_df_3.reset_index(drop=True, inplace=True)
pop_df_3.columns = ['호선', '역명', '인구수']

# =============================================================================
# 6호선 데이터 별도 저장
# =============================================================================
line6_population = pop_df_3.loc[pop_df_3['호선'] == 6, :]
line6_population.reset_index(drop=True, inplace=True)
line6_population.to_csv('line6_population.csv', encoding = 'cp949')

# =============================================================================
# 1~8호선 데이터 별도 처리 start
# =============================================================================
except6_population = pop_df_3.loc[pop_df_3['호선'] != 6, :]
except6_population.reset_index(drop=True, inplace=True)

# 기존 데이터의 역명과 일치하는 데이터 선별작업
except6_population
condi_df = df1.loc[:, ['호선', '역명']]

con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = condi_df.loc[condi_df['호선'] == i, '역명'].values
    v1 = except6_population.loc[except6_population['호선'] == i, '역명'].index
    for j in v1 :
        if except6_population.iloc[j, 1] in condi1 :
            vres.append(j)

# 총 5개의 결측치 발생 => 어디서 뭐가 빠졌는지 확인 필요
condi_df                           # 201행
except6_population.iloc[vres, :]   # 196행


# =============================================================================
# 빠진거 찾아보기
# =============================================================================
# 결론 : 서울역(1), 서울역(4), 성신여대(4), 총신대(4), 이수(7)
except6_population_1 = except6_population.iloc[vres, :]

# 1호선 결측치 : 서울역
len(condi_df.loc[condi_df['호선'] == 1, :])                          # 9
len(except6_population_1.loc[except6_population_1['호선'] == 1,:])   # 8

# 2호선 결측치 : 없음
len(condi_df.loc[condi_df['호선'] == 2, :])                          # 42
len(except6_population_1.loc[except6_population_1['호선'] == 2,:])   # 42

# 3호선 결측치 : 없음
len(condi_df.loc[condi_df['호선'] == 3, :])                          # 27
len(except6_population_1.loc[except6_population_1['호선'] == 3,:])   # 27

# 4호선 결측치 : 서울역, 성신여대, 총신대
len(condi_df.loc[condi_df['호선'] == 4, :])                          # 19
len(except6_population_1.loc[except6_population_1['호선'] == 4,:])   # 16

# 5호선 결측치 : 없음
len(condi_df.loc[condi_df['호선'] == 5, :])                          # 46
len(except6_population_1.loc[except6_population_1['호선'] == 5,:])   # 46

# 7호선 결측치 : 이수
len(condi_df.loc[condi_df['호선'] == 7, :])                          # 41
len(except6_population_1.loc[except6_population_1['호선'] == 7,:])   # 40

# 8호선 결측치 : 없음
len(condi_df.loc[condi_df['호선'] == 8, :])                          # 17
len(except6_population_1.loc[except6_population_1['호선'] == 8,:])   # 17

# =============================================================================
# 결측 데이터 보충
# =============================================================================
# 서울역(1), 서울역(4), 성신여대(4), 총신대(4), 이수(7)
except6_population_2 = pd.concat([except6_population_1, except6_population.loc[except6_population['역명'] == '서울', :]])
except6_population_3 = pd.concat([except6_population_2, except6_population.loc[except6_population['역명'] == '성신여대입구', :]])
except6_population_4 = pd.concat([except6_population_3, except6_population.loc[except6_population['역명'] == '총신대입구(이수)', :]])

# 역명 통일 작업
except6_population_4['역명'] = list(map(name_change, except6_population_4['역명'], except6_population_4['호선']))
def name_change(x, y) :
    if x == '서울' :
        return '서울역'
    elif x == '성신여대입구' :
        return '성신여대'
    elif (x == '총신대입구(이수)') & (y == 4):
        return '총신대'
    elif (x == '총신대입구(이수)') & (y == 7):
        return '이수'
    else :
        return x

except6_population_5 = except6_population_4.sort_values(['호선', '역명'], ascending = [True,True])
except6_population_5.reset_index(drop=True, inplace = True)

# =============================================================================
# 어제의 데이터 주민등록인구수 컬럼 대체 후 모델링
# =============================================================================
df2['주민등록인구수'] = except6_population_5['인구수'].map(lambda x : int(x))

# 분포 확인
df2.hist()      # 변환하면 log 변환이 아니라 다른 스케일링이 필요해 보임
np.log1p(df2).hist()
# 로그변환하면 우측편포

# 1. 주민등록인구수 컬럼 minmax scaling 진행
df2['주민등록인구수'].min()  # 1072
df2['주민등록인구수'].max()  # 175379

df2['주민등록인구수'] = df2['주민등록인구수'].map(lambda x : (x-1072)/(175379-1072))
df2['주민등록인구수'].hist()
# 딱히 분포 변화 없음

# 2. 주민등록인구수 컬럼 standard scaling 진행
mxxn = df2['주민등록인구수'].mean()  # 83057.28855721393
sxd = df2['주민등록인구수'].std()    # 39826.10376607683
df2['주민등록인구수'] = df2['주민등록인구수'].map(lambda x : (x-mxxn)/sxd)
df2['주민등록인구수'].hist()
# 딱히 분포 변화 없음

# 나머지 컬럼은 로그변환
df3 = np.log1p(df2.iloc[:, [1,2,3]])
df3['주민등록인구수'] = df2['주민등록인구수']


# =============================================================================
# train/test 분리
# =============================================================================
from sklearn.model_selection import train_test_split
df3; df_y
train_x, test_x, train_y, test_y = train_test_split(df3.values, df_y, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7466666666666667
m_lr.score(test_x, test_y)             # 0.8431372549019608


# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df3.values):
    train_x, test_x = df3.values[train_index], df3.values[test_index]
    train_y, test_y = df_y[train_index], df_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7462362030905078
np.mean(test_score1)         # 0.7008823529411765


# =============================================================================
# 2023/04/12(수) 추가  - 사업체 데이터 추가
# =============================================================================
company = pd.read_table('D:/03_Python/04_프로젝트/qgis/격자/2020년_사업체_다사_1K.txt', sep='^', header = None)
company = company.drop([0,2],axis=1)
company.columns = ['행정구역', '사업체수']

company_1 = company.groupby('행정구역')[['사업체수']].sum()
company_1 = company_1.reset_index()
company_1.to_csv('cnt_company.csv', encoding='UTF-8')

grid_company = pd.read_csv('grid_company.csv', encoding='UTF-8')
grid_company_1 = grid_company.loc[:, ['호선', '전철역명', 'cnt_company_사업체수']]
grid_company_1

# 중복데이터 제거 / 기존 892개 행에서 657행으로 변환
grid_company_1 = grid_company_1.drop_duplicates(ignore_index = True)

# 조건설정을 통한 1~8호선 데이터만 선택
con1 = ['1', '2', '3', '4', '5', '6', '7', '8']
grid_company_1['호선'] = grid_company_1['호선'].map(lambda x : int(x) if x in con1 else 9)
grid_company_2 = grid_company_1.loc[grid_company_1['호선'] != 9, :]

# 호선, 전철역명 기준 오름차순 정렬
grid_company_3 = grid_company_2.sort_values(['호선', '전철역명'], ascending = [True, True])
grid_company_3.reset_index(drop=True, inplace=True)
grid_company_3.columns = ['호선', '역명', '사업체수']

# =============================================================================
# 6호선 데이터 별도 저장
# =============================================================================
line6_company = grid_company_3.loc[grid_company_3['호선'] == 6, :]
line6_company.reset_index(drop=True, inplace=True)
line6_company.to_csv('line6_company.csv', encoding = 'cp949')

# =============================================================================
# 1~8호선 데이터 별도 처리 start
# =============================================================================
except6_company = grid_company_3.loc[grid_company_3['호선'] != 6, :]
except6_company.reset_index(drop=True, inplace=True)

# 역명 통일 작업
except6_company['역명'] = list(map(name_change, except6_company['역명'], except6_company['호선']))
def name_change(x, y) :
    if x == '서울' :
        return '서울역'
    elif x == '성신여대입구' :
        return '성신여대'
    elif (x == '총신대입구(이수)') & (y == 4):
        return '총신대'
    elif (x == '총신대입구(이수)') & (y == 7):
        return '이수'
    else :
        return x

# 기존 데이터의 역명과 일치하는 데이터 선별작업
except6_company
condi_df = df1.loc[:, ['호선', '역명']]

con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = condi_df.loc[condi_df['호선'] == i, '역명'].values
    v1 = except6_company.loc[except6_company['호선'] == i, '역명'].index
    for j in v1 :
        if except6_company.iloc[j, 1] in condi1 :
            vres.append(j)

# 결측치 : 없음
condi_df                           # 201행
except6_company.iloc[vres, :]      # 201행


# =============================================================================
# 사업체수 결합
# =============================================================================
df2['사업체수'] = except6_company['사업체수']
df4 = np.log1p(df2.iloc[:, 1:])
df4['주민등록인구수'] = df2['주민등록인구수']

except6_company['사업체수'].hist()
df4.hist()


# =============================================================================
# train/test 분리  (df4; df_y 로 진행)
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df4.values, df_y, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.74
m_lr.score(test_x, test_y)             # 0.7450980392156863

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
np.mean(train_score1)        # 0.7479028697571743
np.mean(test_score1)         # 0.6861764705882354


# =============================================================================
# train/test 분리  (df5; df_y 로 진행)
# =============================================================================
df5 = df4.drop('주민등록인구수', axis=1)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df5.values, df_y, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7333333333333333
m_lr.score(test_x, test_y)             # 0.7843137254901961

# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df5.values):
    train_x, test_x = df5.values[train_index], df5.values[test_index]
    train_y, test_y = df_y[train_index], df_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7528366445916115
np.mean(test_score1)         # 0.7059803921568628






# =============================================================================
# 2023/04/12(수) 추가  - 종사자수 데이터 추가
# =============================================================================
worker = pd.read_table('D:/03_Python/04_프로젝트/qgis/격자/2020년_종사자_다사_1K.txt', sep='^', header = None)
worker = worker.drop([0,2],axis=1)
worker.columns = ['행정구역', '종사자수']

worker_1 = worker.groupby('행정구역')[['종사자수']].sum()
worker_1 = worker_1.reset_index()
worker_1.to_csv('cnt_worker.csv', encoding='UTF-8')

grid_worker = pd.read_csv('grid_worker.csv', encoding='UTF-8')
grid_worker_1 = grid_worker.loc[:, ['호선', '전철역명', 'cnt_worker_종사자수']]
grid_worker_1

# 중복데이터 제거 / 기존 892개 행에서 657행으로 변환
grid_worker_1 = grid_worker_1.drop_duplicates(ignore_index = True)

# 조건설정을 통한 1~8호선 데이터만 선택
con1 = ['1', '2', '3', '4', '5', '6', '7', '8']
grid_worker_1['호선'] = grid_worker_1['호선'].map(lambda x : int(x) if x in con1 else 9)
grid_worker_2 = grid_worker_1.loc[grid_company_1['호선'] != 9, :]

# 호선, 전철역명 기준 오름차순 정렬
grid_worker_3 = grid_worker_2.sort_values(['호선', '전철역명'], ascending = [True, True])
grid_worker_3.reset_index(drop=True, inplace=True)
grid_worker_3.columns = ['호선', '역명', '종사자수']

# =============================================================================
# 6호선 데이터 별도 저장
# =============================================================================
line6_worker = grid_worker_3.loc[grid_company_3['호선'] == 6, :]
line6_worker.reset_index(drop=True, inplace=True)
line6_worker.to_csv('line6_worker.csv', encoding = 'cp949')

# =============================================================================
# 1~8호선 데이터 별도 처리 start
# =============================================================================
except6_worker = grid_worker_3.loc[grid_worker_3['호선'] != 6, :]
except6_worker.reset_index(drop=True, inplace=True)

# 역명 통일 작업
grid_worker_3['역명'] = list(map(name_change, grid_worker_3['역명'], grid_worker_3['호선']))
def name_change(x, y) :
    if x == '서울' :
        return '서울역'
    elif x == '성신여대입구' :
        return '성신여대'
    elif (x == '총신대입구(이수)') & (y == 4):
        return '총신대'
    elif (x == '총신대입구(이수)') & (y == 7):
        return '이수'
    else :
        return x

# 기존 데이터의 역명과 일치하는 데이터 선별작업
grid_worker_3
condi_df = df1.loc[:, ['호선', '역명']]

con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = condi_df.loc[condi_df['호선'] == i, '역명'].values
    v1 = grid_worker_3.loc[grid_worker_3['호선'] == i, '역명'].index
    for j in v1 :
        if grid_worker_3.iloc[j, 1] in condi1 :
            vres.append(j)

# 결측치 : 없음
condi_df                           # 201행
grid_worker_3.iloc[vres, :]        # 201행

# =============================================================================
# 종사자수 결합
# =============================================================================
df2['종사자수'] = grid_worker_3['종사자수']
df6 = np.log1p(df2.iloc[:, 1:])
df6['주민등록인구수'] = df2['주민등록인구수']

grid_worker_3['종사자수'].hist()     # 좌측편포
df6.hist()


# =============================================================================
# train/test 분리  (df4; df_y 로 진행)
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df6.values, df_y, random_state=0)

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
for train_index, test_index in m_cv1.split(df6.values):
    train_x, test_x = df6.values[train_index], df6.values[test_index]
    train_y, test_y = df_y[train_index], df_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7512141280353201
np.mean(test_score1)         # 0.6861764705882353



from sklearn.tree import DecisionTreeClassifier as dt_c
m_dt = dt_c()
m_dt.fit(train_x, train_y)
m_dt.score(train_x, train_y)
m_dt.score(test_x, test_y)
m_dt.feature_importances_
# array([0.21578399, 0.30542771, 0.18356416, 0.11809662, 0.07991816,
#        0.09720935])


from sklearn.ensemble import RandomForestRegressor as rf_r
m_rfr = rf_r()
m_rfr.fit(train_x, train_y)
m_rfr.score(train_x, train_y)     # 0.8714989395758304
m_rfr.score(test_x, test_y)       # 0.052224999999999855





# =============================================================================
# grid 데이터 종합 / 종속변수 Y는 2진분류
# =============================================================================
grid_total = pd.concat([sub1, df2], axis=1)
grid_total.to_csv('grid_total.csv', encoding='UTF-8')
pd.read_csv('grid_total.csv', encoding='UTF-8')

total_x = grid_total.iloc[:, 2:8]
total_y = grid_total['등급']

# boxplot 이상치 확인
plt.boxplot(total_x)
plt.xticks(ticks = [1,2,3,4,5,6], labels = total_x.columns)

# 일평균승하차, 사업체수, 종사자수 4분위수 변환
# step 1) 일평균승하차
total_x['일평균승하차'].describe()
Q1 = total_x['일평균승하차'].describe()['25%']
Q3 = total_x['일평균승하차'].describe()['75%']
IQR = Q3 - Q1

# 이상치를 각각 Q1, Q3 값으로 변환
def iqr_function(x) :
    if x < Q1 - (1.5 * IQR) :
        return Q1 - (1.5 * IQR)
    elif x > Q3 + (1.5 * IQR):
        return Q3 + (1.5 * IQR)
    else :
        return x

total_x['일평균승하차'] = total_x['일평균승하차'].map(iqr_function)
plt.boxplot(total_x['일평균승하차'])

# step 2) 사업체수
total_x['사업체수'].describe()
Q1 = total_x['사업체수'].describe()['25%']
Q3 = total_x['사업체수'].describe()['75%']
IQR = Q3 - Q1
total_x['사업체수'] = total_x['사업체수'].map(iqr_function)
plt.boxplot(total_x['사업체수'])

# step 3) 종사자수
total_x['종사자수'].describe()
Q1 = total_x['종사자수'].describe()['25%']
Q3 = total_x['종사자수'].describe()['75%']
IQR = Q3 - Q1
total_x['종사자수'] = total_x['종사자수'].map(iqr_function)
plt.boxplot(total_x['사업체수'])

plt.boxplot(total_x)


# =============================================================================
# 4분위수범위로 이상치 처리 완료
# =============================================================================
total_x.hist()
total_x['주민등록인구수'] = df2['주민등록인구수']
np.log1p(total_x.iloc[:, 1:4]).hist()

total_x.hist()
np.log1p(total_x.iloc[:, 1:4])

# 사업체수, 종사자수 minmax scaling 진행
total_x['사업체수'].min()  # 67
total_x['사업체수'].max()  # 38591

total_x['사업체수'] = total_x['사업체수'].map(lambda x : (x-67)/(38591-67)).hist()
total_x['사업체수'].hist()
np.log1p(total_x['사업체수']).hist()















