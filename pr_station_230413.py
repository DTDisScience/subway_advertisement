# -*- coding: utf-8 -*-
%run my_profile.py

df1 = pd.read_csv('grid_total.csv', encoding='UTF-8')
df1 = df1.drop('Unnamed: 0', axis=1)
df1

floating_pop = pd.read_csv('광고사업진행역_일별_승하차인원_6호선제외.csv', encoding='UTF-8')
floating_pop = floating_pop.loc[floating_pop['승하차인원'] != 0, :]
max_floating = floating_pop.groupby(['호선', '역명'])[['승하차인원']].max()
max_floating_1 = max_floating.reset_index()
max_floating_1
max_floating_1.loc[max_floating_1['역명'] == '개화산', :]

# 학여울(3호선), 동역공(4), 개화산(5), 마장(5), 마천(5), 방화(5), 오금(5), 장암(7)
max_floating_1 = max_floating_1.drop([77,81,101,116,117,123,137,182])
max_floating_1.reset_index(drop=True, inplace=True)

# 1호선 : 동묘앞
df1.loc[df1['호선']== 1, :]                          # 9
max_floating_1.loc[max_floating_1['호선']== 1, :]    # 10

max_floating_1 = max_floating_1.drop(1)
max_floating_1.reset_index(drop=True, inplace = True)

max_floating_1



# =============================================================================
# 출퇴근 시간대로 변경
# 일반적으로 06~09시 까지를 출근 시간대로 봄 => ten to seven도 활성화되는 추세이므로 10시까지 확장
# 퇴근시간대 : 17~20시 사이
# =============================================================================
floating_pop_1 = pd.read_csv('서울교통공사_20221231.csv', encoding='cp949')
floating_pop_2 = floating_pop_1.loc[:, ['수송일자', '호선', '역명', '승하차구분',
                                        '06-07시간대', '07-08시간대', '08-09시간대',
                                        '09-10시간대', '17-18시간대',
                                        '18-19시간대', '19-20시간대']]

# stack 위해 출/퇴근 시간대 나누기
start_pop_1 = floating_pop_2.groupby(['수송일자', '호선', '역명'])[['06-07시간대', '07-08시간대','08-09시간대', '09-10시간대']].sum()
finish_pop_1 = floating_pop_2.groupby(['수송일자', '호선', '역명'])[['17-18시간대','18-19시간대', '19-20시간대']].sum()

start_pop_2 = start_pop_1.stack().reset_index()
start_pop_2.columns = ['수송일자', '호선', '역명', '시간대', '승하차인원']
finish_pop_2 = finish_pop_1.stack().reset_index()
finish_pop_2.columns = ['수송일자', '호선', '역명', '시간대', '승하차인원']

start_pop_3 = start_pop_2.groupby(['수송일자', '호선', '역명'])[['승하차인원']].sum()
start_pop_3 = start_pop_3.reset_index()
finish_pop_3 = finish_pop_2.groupby(['수송일자', '호선', '역명'])[['승하차인원']].sum()
finish_pop_3 = finish_pop_3.reset_index()

max_start_pop = start_pop_3.groupby(['호선', '역명'])[['승하차인원']].max()
max_start_pop = max_start_pop.reset_index()
max_finish_pop = finish_pop_3.groupby(['호선', '역명'])[['승하차인원']].max()
max_finish_pop = max_finish_pop.reset_index()


# =============================================================================
# 출퇴근 시간대 승하차인원
# =============================================================================
max_fl_df = max_start_pop.iloc[:, :2]
max_fl_df['출근시간대'] = max_start_pop['승하차인원']
max_fl_df['퇴근시간대'] = max_finish_pop['승하차인원']
max_fl_df['역명'] = max_fl_df['역명'].map(lambda x : x.split('(')[0])
max_fl_df['역명'] = max_fl_df['역명'].map(lambda x : '성신여대' if x == '성신여대입구' else x)
max_fl_df['역명'] = max_fl_df['역명'].map(lambda x : '총신대' if x == '총신대입구' else x)
max_fl_df

# =============================================================================
#  기존데이터 역명에 해당하는 데이터만 추출하기
# =============================================================================
con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = df1.loc[df1['호선'] == i, '역명'].values
    v1 = max_fl_df.loc[max_fl_df['호선'] == i, '역명'].index
    for j in v1 :
        if max_fl_df.iloc[j, 1] in condi1 :
            vres.append(j)

max_fl_df_1 = max_fl_df.iloc[vres, :]
max_fl_df_1.reset_index(drop=True, inplace = True)

# =============================================================================
# 일평균승하차 => drop / 출퇴근시간대 max 인원 => in
# =============================================================================
df1['출근시간대'] = max_fl_df_1['출근시간대']
df1['퇴근시간대'] = max_fl_df_1['퇴근시간대']

df2 = df1.loc[:, ['호선', '역명', '주민등록인구수', '출근시간대', '퇴근시간대',
                  '대합실면적', '사업체수', '종사자수', '등급']]

df2_x = df2.loc[:, ['출근시간대', '퇴근시간대', '대합실면적', '사업체수', '종사자수']]
df2_y = df2['등급']

df2_x.hist()
np.log1p(df2_x).hist()

df2_x_log = np.log1p(df2_x)

# =============================================================================
# train/test 분리
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2_x_log.values, df2_y, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7333333333333333
m_lr.score(test_x, test_y)             # 0.8235294117647058


# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df2_x_log.values):
    train_x, test_x = df2_x_log.values[train_index], df2_x_log.values[test_index]
    train_y, test_y = df2_y[train_index], df2_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7594922737306843
np.mean(test_score1)         # 0.7208823529411765


# # =============================================================================
# # 출퇴근시간대 => 둘중 max값만 사용하면?
# # =============================================================================
# df2_x = df2.loc[:, ['출근시간대', '퇴근시간대', '대합실면적', '사업체수', '종사자수']]
# df2_y = df2['등급']

# df2_x['max인원'] = list(map(lambda x, y : x if x >= y else y, df1['출근시간대'], df1['퇴근시간대']))

# df2_x_1 = df2_x.iloc[:, 2:]
# df2_x_1.hist()

# df2_x_1_log = np.log1p(df2_x_1)

# # =============================================================================
# # train/test 분리
# # =============================================================================
# from sklearn.model_selection import train_test_split
# train_x, test_x, train_y, test_y = train_test_split(df2_x_1_log.values, df2_y, random_state=0)

# # =============================================================================
# # 로지스틱 회귀 모델링
# # =============================================================================
# from sklearn.linear_model import LogisticRegression as lr
# m_lr = lr()
# m_lr.fit(train_x, train_y)
# m_lr.score(train_x, train_y)           # 0.72
# m_lr.score(test_x, test_y)             # 0.7843137254901961

# # =============================================================================
# # K-Fold 교차검증
# # =============================================================================
# from sklearn.model_selection import KFold
# m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

# train_score1 = []; test_score1 = []
# for train_index, test_index in m_cv1.split(df2_x_1_log.values):
#     train_x, test_x = df2_x_1_log.values[train_index], df2_x_1_log.values[test_index]
#     train_y, test_y = df2_y[train_index], df2_y[test_index]

#     m_lr = lr()
#     m_lr.fit(train_x, train_y)
    
#     train_score1.append(m_lr.score(train_x, train_y))
#     test_score1.append(m_lr.score(test_x, test_y))

# # 교차검증 후 최종 점수
# np.mean(train_score1)        # 0.7428918322295806
# np.mean(test_score1)         # 0.6912745098039215


# # =============================================================================
# # 출퇴근시간대 => 둘을 합치면?
# # =============================================================================
# df2_x = df2.loc[:, ['출근시간대', '퇴근시간대', '대합실면적', '사업체수', '종사자수']]
# df2_y = df2['등급']

# df2_x['max인원'] = list(map(lambda x, y : x + y, df1['출근시간대'], df1['퇴근시간대']))

# df2_x_2 = df2_x.iloc[:, 2:]
# df2_x_2.hist()

# df2_x_2_log = np.log1p(df2_x_2)

# # =============================================================================
# # train/test 분리
# # =============================================================================
# from sklearn.model_selection import train_test_split
# train_x, test_x, train_y, test_y = train_test_split(df2_x_2_log.values, df2_y, random_state=0)

# # =============================================================================
# # 로지스틱 회귀 모델링
# # =============================================================================
# from sklearn.linear_model import LogisticRegression as lr
# m_lr = lr()
# m_lr.fit(train_x, train_y)
# m_lr.score(train_x, train_y)           # 0.72
# m_lr.score(test_x, test_y)             # 0.7843137254901961

# # =============================================================================
# # K-Fold 교차검증
# # =============================================================================
# from sklearn.model_selection import KFold
# m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

# train_score1 = []; test_score1 = []
# for train_index, test_index in m_cv1.split(df2_x_2_log.values):
#     train_x, test_x = df2_x_2_log.values[train_index], df2_x_2_log.values[test_index]
#     train_y, test_y = df2_y[train_index], df2_y[test_index]

#     m_lr = lr()
#     m_lr.fit(train_x, train_y)
    
#     train_score1.append(m_lr.score(train_x, train_y))
#     test_score1.append(m_lr.score(test_x, test_y))

# # 교차검증 후 최종 점수
# np.mean(train_score1)        # 0.739580573951435
# np.mean(test_score1)         # 0.7011764705882353


# =============================================================================
# 
# =============================================================================
df2_x = df2.loc[:, ['주민등록인구수', '출근시간대', '퇴근시간대', '대합실면적', '사업체수', '종사자수']]
df2_y = df2['등급']

plt.figure()
df2_x['주민등록인구수'].hist()


single_grid_df = pd.read_csv('230413_단일_grid.csv', encoding='UTF-8')
single_grid_df = single_grid_df.loc[:, ['호선', '전철역명', '격자인구_3', 'cnt_company_사업체수', 'cnt_worker_종사자수']]
single_grid_df.columns = ['호선', '역명', '인구수', '사업체수', '종사자수']

con1 = ['1', '2', '3', '4', '5', '6', '7', '8']
single_grid_df['호선'] = single_grid_df['호선'].map(lambda x : int(x) if x in con1 else 9)
single_grid_df_1 = single_grid_df.loc[single_grid_df['호선'] != 9, :]
single_grid_df_1.reset_index(drop=True, inplace=True)

single_grid_df_1 = single_grid_df_1.drop_duplicates()
single_grid_df_1 = single_grid_df_1.sort_values(['호선', '역명'], ascending = [True, True])
single_grid_df_1.reset_index(drop=True, inplace=True)
single_grid_df_1

# =============================================================================
# 6호선 데이터 별도 저장
# =============================================================================
line6_single_grid = single_grid_df_1.loc[single_grid_df_1['호선'] == 6, :]
line6_single_grid.reset_index(drop=True, inplace=True)
line6_single_grid.to_csv('line6_single_grid.csv', encoding = 'cp949')

# =============================================================================
# 1~8호선 데이터 별도 처리 start
# =============================================================================
except6_single_grid = single_grid_df_1.loc[single_grid_df_1['호선'] != 6, :]
except6_single_grid.reset_index(drop=True, inplace=True)

# 역명 통일 작업
except6_single_grid['역명'] = list(map(name_change, except6_single_grid['역명'], except6_single_grid['호선']))
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
except6_single_grid
condi_df = df1.loc[:, ['호선', '역명']]

con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = condi_df.loc[condi_df['호선'] == i, '역명'].values
    v1 = except6_single_grid.loc[except6_single_grid['호선'] == i, '역명'].index
    for j in v1 :
        if except6_single_grid.iloc[j, 1] in condi1 :
            vres.append(j)

# 결측치 : 없음
condi_df                               # 201행
except6_single_grid.iloc[vres, :]      # 201행

# =============================================================================
# 데이터프레임 결합
# =============================================================================
df2 = df1.loc[:, ['호선', '역명', '주민등록인구수', '출근시간대', '퇴근시간대',
                  '대합실면적', '사업체수', '종사자수', '등급']]
except6_single_grid = except6_single_grid.iloc[vres, :]
except6_single_grid.reset_index(drop=True, inplace=True)

except6_single_grid['인구수'].sum()
except6_single_grid['사업체수'].sum()

df2_x_3 = except6_single_grid.iloc[:, 2:]
df2_x_3['출근시간대'] = df2['출근시간대']
df2_x_3['퇴근시간대'] = df2['퇴근시간대']
df2_x_3['대합실면적'] = df2['대합실면적']

df2_x_3.hist()


df2_x_3['인구수'].min()  # 278
df2_x_3['인구수'].max()  # 43853
df2_x_3['인구수'] = df2_x_3['인구수'].map(lambda x : (x-278)/(43853-278))
df2_x_3['사업체수'] = np.log1p(df2_x_3['사업체수'])
df2_x_3['종사자수'] = np.log1p(df2_x_3['종사자수'])
df2_x_3['출근시간대'] = np.log1p(df2_x_3['출근시간대'])
df2_x_3['퇴근시간대'] = np.log1p(df2_x_3['퇴근시간대'])
df2_x_3['대합실면적'] = np.log1p(df2_x_3['대합실면적'])


# =============================================================================
# train/test 분리  (df4; df_y 로 진행)
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2_x_3.values, df2_y, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.76
m_lr.score(test_x, test_y)             # 0.803921568627451

# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df2_x_3.values):
    train_x, test_x = df2_x_3.values[train_index], df2_x_3.values[test_index]
    train_y, test_y = df2_y[train_index], df2_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.76280353200883
np.mean(test_score1)         # 0.7162745098039216



company = pd.read_table('사업체수_수정.txt', encoding='UTF-8', sep='^', header=None)
company.columns = ['년도', '행정구역', '코드', '사업체수']
cnt1 = company.groupby('행정구역')[['사업체수']].sum()
cnt1 = cnt1.reset_index()
cnt1['사업체수'].sum()


# =============================================================================
# 
# =============================================================================
dong = pd.read_table('행정구역코드통합_1.txt', sep=',')
dong = dong.drop('인덱스', axis=1)
dong['지역코드'] = dong['지역코드'].map(lambda x : x.split('(')[0])

dong_code = pd.read_clipboard()
dong_code['시군구코드'] = dong_code['시군구코드'].map(lambda x : '0' + str(x) if len(str(x)) == 2 else str(x))
dong_code = dong_code.drop(['시도명칭', '시군구명칭'], axis=1)
dong_code['코드'] = list(map(lambda x, y, z : str(x) + str(y) + str(z), dong_code['시도코드'], dong_code['시군구코드'], dong_code['읍면동코드']))
dong_code_1 = dong_code.iloc[:, 3:]

dong_code_1
dong_code_1.loc[dong_code_1['코드'] == '11250700', '읍면동명칭']
vres1 = []
for i in list(dong['지역코드'].values):
    if i[0] == '1' :
        vres1.append(dong_code_1.loc[dong_code_1['코드'] == i, '읍면동명칭'].values[0])
    else :
        vres1.append(dong.loc[dong['지역코드'] == i, '행정동'].values[0])

dong.loc[dong['지역코드'] == '1104068', :]

len(vres1)
dong['행정동수정'] = vres1
dong

dong_company = pd.read_csv('동별사업체수.csv')
dong_company = dong_company.drop('Unnamed: 0', axis=1)
dong_company.columns = ['동명', '사업체수']

dong_company['동명'] = dong_company['동명'].map(lambda x : x.replace('.', '·'))

dong['사업체수']
vres2 = []
for i in list(dong['행정동수정'].values):
    condi2 = dong_company['동명'].values
    if i in condi2 :
        vres2.append(dong_company.loc[dong_company['동명'] == str(i), '사업체수'].values[0])
    else :
        vres2.append('없음')

dong['사업체수'] = vres2
dong.loc[dong['사업체수'] =='없음', :]
# 156   7   광명사거리   광명동  31060530   광명동   없음
# 183   7      장암   장암동  31030560   장암동   없음
# 188   7      철산   철산동  31060600   철산동   없음
# 195   8  남한산성입구   단대동  31021600   단대동   없음
# 196   8   단대오거리   신흥동  31021520   신흥동   없음
# 197   8      모란   성남동  31021590   성남동   없음
# 201   8      산성   신흥동  31021520   신흥동   없음
# 204   8      수진  수진1동  31021580  수진1동   없음

res1 = [4963, 383, 6236, 2873, 5380, 7183, 2873, 6146]
st_name = ['광명사거리', '장암', '철산', '남한산성입구', '단대오거리', '모란', '산성', '수진']
for i in range(0, len(st_name)) :
    dong.loc[dong['역명'] == st_name[i], '사업체수'] = res1[i]

dong_1 = dong.drop([77,78,82,102,117,118,124,138,183])
dong_1.reset_index(drop=True, inplace=True)
dong_1 = dong_1.drop(['행정동', '지역코드'], axis = 1)

dong_1

df2_x['사업체수'] = dong_1['사업체수'].map(lambda x : int(x))

df2_x_log1 = np.log1p(df2_x)
df2_x_log1.hist()


# =============================================================================
# train/test 분리
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2_x_log1.values, df2_y, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7066666666666667
m_lr.score(test_x, test_y)             # 0.8431372549019608


# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df2_x_log1.values):
    train_x, test_x = df2_x_log1.values[train_index], df2_x_log1.values[test_index]
    train_y, test_y = df2_y[train_index], df2_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7495033112582781
np.mean(test_score1)         # 0.6859803921568628


from sklearn.tree import DecisionTreeClassifier as dt_c
m_dt = dt_c()
m_dt.fit(train_x, train_y)
m_dt.score(train_x, train_y)
m_dt.score(test_x, test_y)
m_dt.feature_importances_
# array([0.26118797, 0.34075021, 0.29590152, 0.07496943, 0.02719088])
#       ['출근시간대', '퇴근시간대', '대합실면적', '사업체수', '종사자수']
df2_x_log1.columns




# =============================================================================
# 230413(목) 스타벅스 데이터 결합
# =============================================================================
bucks = pd.read_csv('스타벅스수500m.csv')
bucks = bucks.drop('Unnamed: 0', axis = 1)
bucks_1 = bucks.loc[bucks['호선'] != 6, :]
bucks_1.reset_index(drop=True, inplace=True)

con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = condi_df.loc[condi_df['호선'] == i, '역명'].values
    v1 = bucks_1.loc[bucks_1['호선'] == i, '역명'].index
    for j in v1 :
        if bucks_1.iloc[j, 0] in condi1 :
            vres.append(j)
len(vres)
bucks_1['역명'] = list(map(name_change, bucks_1['역명'], bucks_1['호선']))
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

bucks_2 =  bucks_1.iloc[vres, :]
bucks_2.reset_index(drop=True, inplace=True)

df2_x['스타벅스수'] = bucks_2['스타벅스수']
df2_x.hist()

df2_x1 = df2_x.iloc[:, [0,1,2,3, 5]]
df2_x1_log = np.log1p(df2_x1)

df2_x
df2_x1_log.hist()
df2_x['스타벅스수'].hist()



# =============================================================================
# train/test 분리
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2_x1_log.values, df2_y, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7066666666666667
m_lr.score(test_x, test_y)             # 0.8431372549019608


# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df2_x1_log.values):
    train_x, test_x = df2_x1_log.values[train_index], df2_x1_log.values[test_index]
    train_y, test_y = df2_y[train_index], df2_y[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7694260485651214
np.mean(test_score1)         # 0.7455882352941177

df2_x.to_csv('스타벅스수포함_230413.csv', encoding='UTF-8')

(df2_y == 1).sum()  69
(df2_y == 2).sum() 132

# =============================================================================
# 3진분류
# =============================================================================
grade_1 = pd.read_csv('subway1_8.csv')


grade_1 = grade_1.drop(77)
grade_1.reset_index(drop=True, inplace=True)
grade_1 = grade_1.drop([77, 81,101,116,117,123,137,182])
grade_1.reset_index(drop=True, inplace=True)
grade_1['등급'] = grade_1['등급'].map(lambda x : int(x)).map(lambda x : 1 if x == 2 else x).map(lambda x : 2 if x == 3 else x).map(lambda x : 3 if x == 4 else x)
grade_1['등급'] = grade_1['등급'].astype('category')
grade_1['급지분류'] = grade_1['급지분류'].astype('category')


from sklearn.tree import DecisionTreeClassifier as dt_c
m_dt = dt_c()
m_dt.fit(train_x, train_y)
m_dt.score(train_x, train_y)
m_dt.score(test_x, test_y)
m_dt.feature_importances_
# array([0.17488284, 0.19641809, 0.26840362, 0.17507352, 0.18522193])
#       ['출근시간대', '퇴근시간대', '대합실면적', '사업체수', '스타벅스수']
df2_x1_log.columns



# =============================================================================
# train/test 분리
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2_x1_log.values, grade_1['급지분류'].values, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7066666666666667
m_lr.score(test_x, test_y)             # 0.8431372549019608

coefficients = pd.DataFrame({"Feature":df2_x1_log.columns, "Coefficients":np.transpose(m_lr.coef_[0])})
coefficients = coefficients.sort_values(by='Coefficients', ascending=False)

#       Feature  Coefficients
# 1   퇴근시간대      0.823432
# 4   스타벅스수      0.579303
# 2   대합실면적      0.143817
# 3    사업체수      0.003734
# 0   출근시간대     -0.062684
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import SequentialFeatureSelector

sfs = SequentialFeatureSelector(m_lr, k_features='best', forward=False, verbose=2, scoring='accuracy', cv=5)
sfs.fit(df2_x1_log.values, df2_y)

# 결과 출력
print("선택된 변수: ", sfs.k_feature_names_)
df2_x1_log.columns
print("검증 정확도: ", sfs.k_score_)



df2_x3_log = df2_x1_log.iloc[:, [1,2,4]]


# =============================================================================
# train/test 분리
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2_x3_log.values, df2_y.values, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.74
m_lr.score(test_x, test_y)             # 0.8627450980392157


# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df2_x3_log.values):
    train_x, test_x = df2_x3_log.values[train_index], df2_x3_log.values[test_index]
    train_y, test_y = df2_y.values[train_index], df2_y.values[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7776931567328919
np.mean(test_score1)         # 0.7653921568627451


df2_x3_log; df2_y

from sklearn.ensemble import RandomForestClassifier as rf_c
m_rfc = rf_c()
m_rfc.fit(train_x, train_y)
m_rfc.score(train_x, train_y)     # 1.0
m_rfc.score(test_x, test_y)       # 0.7450980392156863
m_rfc.feature_importances_

from sklearn.model_selection import cross_val_score, train_test_split
scores = cross_val_score(m_rfc, train_x, train_y, cv=5)

# 교차검증 결과를 출력합니다.
print("Cross-validation scores: {}".format(scores))
print("Average score: {:.2f}".format(scores.mean()))



from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2_x1_log.values, df2_y.values, random_state=0)

from sklearn.ensemble import RandomForestClassifier as rf_c
m_rfc = rf_c()
m_rfc.fit(train_x, train_y)
m_rfc.score(train_x, train_y)     # 1.0
m_rfc.score(test_x, test_y)       # 0.7647058823529411
m_rfc.feature_importances_
# [0.17114952, 0.22007273, 0.28188328, 0.16889182, 0.15800265]
# ['출근시간대', '퇴근시간대', '대합실면적', '사업체수', '스타벅스수']

sfs = SequentialFeatureSelector(m_rfc, k_features='best', forward=False, verbose=2, scoring='accuracy', cv=5)
sfs.fit(df2_x1_log.values, df2_y)

# 결과 출력
print("선택된 변수: ", sfs.k_feature_names_)
# ('0', '1', '2', '4')
# ['출근시간대', '퇴근시간대', '대합실면적', '스타벅스수']
print("검증 정확도: ", sfs.k_score_)
# 0.785609756097561

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2_x1_log.iloc[:, [0,1,2,4]].values, df2_y.values, random_state=0)
from sklearn.ensemble import RandomForestClassifier as rf_c
m_rfc = rf_c()
m_rfc.fit(train_x, train_y)
m_rfc.score(train_x, train_y)     # 1.0
m_rfc.score(test_x, test_y)       # 0.8235294117647058
m_rfc.feature_importances_

# =============================================================================
# K-Fold 교차검증
# =============================================================================
from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df2_x3_log.values):
    train_x, test_x = df2_x1_log.iloc[:, [0,1,2,4]].values[train_index], df2_x1_log.iloc[:, [0,1,2,4]].values[test_index]
    train_y, test_y = df2_y.values[train_index], df2_y.values[test_index]

    m_rfc = rf_c()
    m_rfc.fit(train_x, train_y)
    
    train_score1.append(m_rfc.score(train_x, train_y))
    test_score1.append(m_rfc.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 1.0
np.mean(test_score1)         # 0.7607843137254902



