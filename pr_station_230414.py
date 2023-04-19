# -*- coding: utf-8 -*-
%run my_profile.py

df1 = pd.read_csv('grid_total.csv', encoding='UTF-8', index_col=0)
condi_df = df1.loc[:, ['호선', '역명']]

df_230413 = pd.read_csv('스타벅스수포함_230413.csv', encoding='UTF-8', index_col=0)
df_230413_1 = df_230413.iloc[:, [0,1,2,5]]
df_230413_1; condi_df

df_230413_2 = pd.concat([condi_df, df_230413_1], axis=1)
df_230413_2

# 승하차 인원 데이터 로딩
floating_pop_1 = pd.read_csv('서울교통공사_20221231.csv', encoding='cp949')
# 필요한 컬럼 추출
floating_pop_2 = floating_pop_1.loc[:, ['수송일자', '호선', '역명', '승하차구분',
                                        '06-07시간대', '07-08시간대', '08-09시간대',
                                        '09-10시간대', '17-18시간대',
                                        '18-19시간대', '19-20시간대']]
# 승차 / 하차 데이터 구분
floating_pop_in = floating_pop_2.loc[floating_pop_2['승하차구분'] == '승차', :]
floating_pop_out = floating_pop_2.loc[floating_pop_2['승하차구분'] == '하차', :]

# 승차인원 기준 / 출퇴근 시간대 구분
in_start_1 = floating_pop_in.groupby(['수송일자', '호선', '역명'])[['06-07시간대', '07-08시간대','08-09시간대', '09-10시간대']].sum()
in_finish_1 = floating_pop_in.groupby(['수송일자', '호선', '역명'])[['17-18시간대','18-19시간대', '19-20시간대']].sum()

# sum을 위한 stack (wide → long)
in_start_2 = in_start_1.stack().reset_index()
in_start_2.columns = ['수송일자', '호선', '역명', '출근시간대', '승차인원']
in_finish_2 = in_finish_1.stack().reset_index()
in_finish_2.columns = ['수송일자', '호선', '역명', '퇴근시간대', '승차인원']

# 그룹합계 실행
in_start_3 = in_start_2.groupby(['수송일자', '호선', '역명'])[['승차인원']].sum()
in_start_3 = in_start_3.reset_index()
in_finish_3 = in_finish_2.groupby(['수송일자', '호선', '역명'])[['승차인원']].sum()
in_finish_3 = in_finish_3.reset_index()

# 2022년 호선, 역명 기준 출퇴근 시간대 max 승차인원
max_in_start = in_start_3.groupby(['호선', '역명'])[['승차인원']].max()
max_in_start = max_in_start.reset_index()
max_in_start.columns = ['호선', '역명', '출근시간대승차인원']
max_in_finish = in_finish_3.groupby(['호선', '역명'])[['승차인원']].max()
max_in_finish = max_in_finish.reset_index()
max_in_finish.columns = ['호선', '역명', '퇴근시간대승차인원']

# =============================================================================
# 하차인원 기준 / 출퇴근 시간대 구분
out_start_1 = floating_pop_out.groupby(['수송일자', '호선', '역명'])[['06-07시간대', '07-08시간대','08-09시간대', '09-10시간대']].sum()
out_finish_1 = floating_pop_out.groupby(['수송일자', '호선', '역명'])[['17-18시간대','18-19시간대', '19-20시간대']].sum()

# sum을 위한 stack (wide → long)
out_start_2 = out_start_1.stack().reset_index()
out_start_2.columns = ['수송일자', '호선', '역명', '출근시간대', '하차인원']
out_finish_2 = out_finish_1.stack().reset_index()
out_finish_2.columns = ['수송일자', '호선', '역명', '퇴근시간대', '하차인원']

# 그룹합계 실행
out_start_3 = out_start_2.groupby(['수송일자', '호선', '역명'])[['하차인원']].sum()
out_start_3 = out_start_3.reset_index()
out_finish_3 = out_finish_2.groupby(['수송일자', '호선', '역명'])[['하차인원']].sum()
out_finish_3 = out_finish_3.reset_index()

# 2022년 호선, 역명 기준 출퇴근 시간대 max 승차인원
max_out_start = out_start_3.groupby(['호선', '역명'])[['하차인원']].max()
max_out_start = max_out_start.reset_index()
max_out_start.columns = ['호선', '역명', '출근시간대하차인원']
max_out_finish = out_finish_3.groupby(['호선', '역명'])[['하차인원']].max()
max_out_finish = max_out_finish.reset_index()
max_out_finish.columns = ['호선', '역명', '퇴근시간대하차인원']

# =============================================================================
# 출근시간대승차인원, 퇴근시간대승차인원, 출근시간대하차인원, 퇴근시간대하차인원 컬럼 합치기
# =============================================================================
df_inout = max_in_start[:]
df_inout['퇴근시간대승차인원'] = max_in_finish['퇴근시간대승차인원']
df_inout['출근시간대하차인원'] = max_out_start['출근시간대하차인원']
df_inout['퇴근시간대하차인원'] = max_out_finish['퇴근시간대하차인원']

df_inout


# =============================================================================
# 기존 데이터 역명에 해당하는 행만 추출하기
# =============================================================================
# 1. df_inout 역명 컬럼 통일화 작업
df_inout['역명'] = df_inout['역명'].map(lambda x : x.split('(')[0])
df_inout['역명'] = df_inout['역명'].map(lambda x : '성신여대' if x == '성신여대입구' else x)
df_inout['역명'] = df_inout['역명'].map(lambda x : '총신대' if x == '총신대입구' else x)

# 2. 추출하기
con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = df1.loc[df1['호선'] == i, '역명'].values
    v1 = df_inout.loc[df_inout['호선'] == i, '역명'].index
    for j in v1 :
        if df_inout.iloc[j, 1] in condi1 :
            vres.append(j)

df_inout_result = df_inout.iloc[vres, :]
df_inout_result.reset_index(drop=True, inplace = True)

# 3. 결과물 확인
df_inout_result


# =============================================================================
# df_230413_2; df_inout_result 로 모델링
# =============================================================================
df_230413_2
df_230413_3 = df_230413_2[:]

# 1. 컬럼 합치기
df_230413_3['출근시간대승차인원'] = df_inout_result['출근시간대승차인원']
df_230413_3['퇴근시간대승차인원'] = df_inout_result['퇴근시간대승차인원']
df_230413_3['출근시간대하차인원'] = df_inout_result['출근시간대하차인원']
df_230413_3['퇴근시간대하차인원'] = df_inout_result['퇴근시간대하차인원']

df_230413_3

# 2. 독립변수/종속변수 분리
df_x_230413 = df_230413_3.iloc[:, 2:]
df_y_230413 = df1.iloc[:, 8].values

# 3. 독립변수 로그변환
df_x_log_230413 = np.log1p(df_x_230413.values)

# 4. 필요모듈 import
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier as rf_c
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import KFold
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance

# 5. train/test 분리
train_x, test_x, train_y, test_y = train_test_split(df_x_log_230413, df_y_230413, random_state=0)

# 6. 로지스틱 회귀 모델링
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.76
m_lr.score(test_x, test_y)             # 0.8627450980392157

# 7. SequentialFeatureSelector로 최적독립변수 조합 확인
from mlxtend.feature_selection import SequentialFeatureSelector
sfs = SequentialFeatureSelector(m_lr, k_features='best', forward=False, verbose=2, scoring='accuracy', cv=4)
sfs.fit(df_x_log_230413, df_y_230413)
sfs.
# 8. 결과 출력
print("선택된 변수: ", sfs.k_feature_names_)
df_x_230413.columns[[0, 2, 3, 5, 6]]
print("검증 정확도: ", sfs.k_score_)
# 선택된 변수:  ('0', '2', '3', '5', '6')
# ['출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원']
# 검증 정확도:  0.7756862745098039


# =============================================================================
# 
# =============================================================================
train_x, test_x, train_y, test_y = train_test_split(df_x_log_230413[:, [0,2,3,5,6]], df_y_230413, random_state=0)
m_lr = lr()
result = m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7533333333333333
m_lr.score(test_x, test_y)             # 0.8627450980392157

from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df_x_log_230413[:, [0,2,3,5,6]]):
    train_x, test_x = df_x_log_230413[:, [0,2,3,5,6]][train_index], df_x_log_230413[:, [0,2,3,5,6]][test_index]
    train_y, test_y = df_y_230413[train_index], df_y_230413[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7677593818984547
np.mean(test_score1)         # 0.7604901960784314

# 최종모델 변수 중요도 파악하기
results = permutation_importance(m_lr, df_x_log_230413[:, [0,2,3,5,6]], df_y_230413, n_repeats=10, random_state=0)
# 결과 시각화
importance = results.importances_mean
feature_names = df_x_230413.iloc[:, [0,2,3,5,6]].columns
sorted_idx = importance.argsort()

plt.barh(range(len(sorted_idx)), importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Permutation Importance (Logistic Regression)")
plt.show()


# =============================================================================
# 9. RandomForestClassifier 모델링
m_rfc = rf_c()
m_rfc.fit(train_x, train_y)
m_rfc.score(train_x, train_y)     # 1.0
m_rfc.score(test_x, test_y)       # 0.8235294117647058

# 10. SequentialFeatureSelector로 최적독립변수 조합 확인
sfs = SequentialFeatureSelector(m_rfc, k_features='best', forward=False, verbose=2, scoring='accuracy', cv=4)
sfs.fit(train_x, train_y)

# 11. 결과 출력
print("선택된 변수: ", sfs.k_feature_names_)
df_x_230413.columns[[0, 1, 2, 5, 6, 7]]
print("검증 정확도: ", sfs.k_score_)
# 선택된 변수:  ('0', '1', '2', '5', '6', '7')
# ['출근시간대', '퇴근시간대', '대합실면적', '퇴근시간대승차인원', '출근시간대하차인원', '퇴근시간대하차인원']
# 검증 정확도:  0.8001422475106686


# =============================================================================
# =============================================================================
# =============================================================================
# 승객유형 데이터 가공
# =============================================================================
passenger_type = pd.read_csv('서울교통공사_1-8호선 역별 일별 시간대별 승객유형별 승하차인원_20220630.csv', encoding='cp949')
passenger_type = passenger_type.loc[:, ['수송일자', '호선', '역명', '승하차구분',
                                        '승객유형', '06-07시간대', '07-08시간대', '08-09시간대',
                                        '09-10시간대', '17-18시간대',
                                        '18-19시간대', '19-20시간대']]

passenger_type['승객유형'] = passenger_type['승객유형'].map(type_function)
def type_function(x):
    if x in ['어린이', '중고생'] :
        return '청소년'
    elif x in ['영어 일반', '일어 일반', '중국어 일반', '영어 어린이', '일어 어린이', '중국어 어린이'] :
        return '외국인'
    elif x == '우대권' :
        return '우대권'
    else :
        return '일반'

df_x_230413 


# 승차 기준 퇴근시간대 일반인탑승비율 구하기
type_in_1 = passenger_type.loc[passenger_type['승하차구분'] == '승차', :]
type_in_2 = type_in_1.iloc[:, [0,1,2,4,9, 10, 11]]
type_in_3 = type_in_2.groupby(['수송일자', '호선', '역명', '승객유형'])[['17-18시간대', '18-19시간대', '19-20시간대']].sum()
type_in_3 = type_in_3.stack().reset_index()
type_in_3.columns = ['수송일자', '호선', '역명', '승객유형', '시간대', '출근시간대하차인원']
type_in_4 = type_in_3.groupby(['수송일자','호선', '역명', '승객유형'])[['출근시간대하차인원']].sum()
type_in_5 = type_in_4.unstack(level = -1, fill_value = 0)
type_in_5 = type_in_5.reset_index()
type_in_5.columns = ['수송일자','호선', '역명', '외국인', '우대권', '일반', '청소년']
type_in_5['합계'] = type_in_5.sum(axis=1).values
type_in_5['일반인비율'] = list(map(lambda x, y : x/y, type_in_5['일반'], type_in_5['합계']))
type_in_6 = type_in_5.groupby(['호선', '역명'])[['일반인비율']].mean()
type_in_6 = type_in_6.reset_index()

# 1. type_in_6 역명 컬럼 통일화 작업
type_in_6['역명'] = type_in_6['역명'].map(lambda x : x.split('(')[0])
type_in_6['역명'] = type_in_6['역명'].map(lambda x : '성신여대' if x == '성신여대입구' else x)
type_in_6['역명'] = type_in_6['역명'].map(lambda x : '총신대' if x == '총신대입구' else x)

# 2. 추출하기
con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = df1.loc[df1['호선'] == i, '역명'].values
    v1 = type_in_6.loc[type_in_6['호선'] == i, '역명'].index
    for j in v1 :
        if type_in_6.iloc[j, 1] in condi1 :
            vres.append(j)

type_in_finish_result = type_in_6.iloc[vres, :]
type_in_finish_result.reset_index(drop=True, inplace = True)

# 3. 결과물 확인
type_in_finish_result

# =============================================================================
# 승차 기준 퇴근시간대 일반인탑승비율 구하기
# =============================================================================
type_out_1 = passenger_type.loc[passenger_type['승하차구분'] == '승차', :]
type_out_2 = type_out_1.iloc[:, :9] 
type_out_3 = type_out_2.groupby(['수송일자', '호선', '역명', '승객유형'])[['06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대']].sum()
type_out_3 = type_out_3.stack().reset_index()
type_out_3.columns = ['수송일자', '호선', '역명', '승객유형', '시간대', '퇴근시간대승차인원']
type_out_4 = type_out_3.groupby(['수송일자','호선', '역명', '승객유형'])[['퇴근시간대승차인원']].sum()
type_out_5 = type_out_4.unstack(level = -1, fill_value = 0)
type_out_5 = type_out_5.reset_index()
type_out_5.columns = ['수송일자','호선', '역명', '외국인', '우대권', '일반', '청소년']
type_out_5['합계'] = type_out_5.sum(axis=1).values
type_out_5['일반인비율'] = list(map(lambda x, y : x/y, type_out_5['일반'], type_out_5['합계']))
type_out_6 = type_out_5.groupby(['호선', '역명'])[['일반인비율']].mean()
type_out_6 = type_out_6.reset_index()

# 1. type_in_6 역명 컬럼 통일화 작업
type_out_6['역명'] = type_out_6['역명'].map(lambda x : x.split('(')[0])
type_out_6['역명'] = type_out_6['역명'].map(lambda x : '성신여대' if x == '성신여대입구' else x)
type_out_6['역명'] = type_out_6['역명'].map(lambda x : '총신대' if x == '총신대입구' else x)

# 2. 추출하기
con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = df1.loc[df1['호선'] == i, '역명'].values
    v1 = type_out_6.loc[type_out_6['호선'] == i, '역명'].index
    for j in v1 :
        if type_out_6.iloc[j, 1] in condi1 :
            vres.append(j)

type_out_start_result = type_out_6.iloc[vres, :]
type_out_start_result.reset_index(drop=True, inplace = True)

# 3. 결과물 확인
type_out_start_result

# =============================================================================
# 원본데이터와 결합
# =============================================================================
df_x_230413['출근시간대일반인하차'] = list(map(lambda x, y : round(x * y), df_x_230413['출근시간대승차인원'], type_out_start_result['일반인비율']))
df_x_230413['퇴근시간대일반인승차'] = list(map(lambda x, y : round(x * y), df_x_230413['퇴근시간대승차인원'], type_in_finish_result['일반인비율']))

df_x_230413.hist()
plt.boxplot(df_x_230413)

df_x_log1_230413 = np.log1p(df_x_230413).values
train_x, test_x, train_y, test_y = train_test_split(df_x_log1_230413, df_y_230413, random_state=0)

# 7. SequentialFeatureSelector로 최적독립변수 조합 확인
from sklearn.linear_model import LogisticRegression as lr
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SequentialFeatureSelector
m_lr = lr()
sfs = SequentialFeatureSelector(m_lr, k_features='best', forward=False, verbose=2, scoring='accuracy', cv=4)
sfs = SequentialFeatureSelector(m_lr, n_features_to_select=5, cv=4)
sfs.fit(df_x_log1_230413, df_y_230413)
selected_mask = sfs.get_support()
selected_features = df_x_230413.columns[selected_mask].tolist()
removed_features = df_x_230413.columns[~selected_mask].tolist()
print("Selected features:", selected_features)
# ['퇴근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '퇴근시간대일반인승차']
print("Removed features:", removed_features)
# ['출근시간대', '출근시간대승차인원', '출근시간대하차인원', '퇴근시간대하차인원', '출근시간대일반인하차']

# 8. 결과 출력
print("선택된 변수: ", sfs.k_feature_names_)
df_x_230413.columns[[0, 2, 3, 5, 6]]
print("검증 정확도: ", sfs.k_score_)
# 선택된 변수:  ('0', '2', '3', '5', '6')
# ['출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원']
# 검증 정확도:  0.7756862745098039
sfs.get_support()
sfs.best_estimator_.get_support()
sfs.best_estimator_

# =============================================================================
df_x_final = df_x_230413.loc[:, ['대합실면적', '스타벅스수', '퇴근시간대승차인원', '퇴근시간대일반인승차']]
df_x_final = np.log1p(df_x_final).values
train_x, test_x, train_y, test_y = train_test_split(df_x_final, df_y_230413, random_state=0)
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)        # 0.74
m_lr.score(test_x, test_y)          # 0.8627450980392157

m_cv1 = KFold(4)
train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df_x_final):
    train_x, test_x = df_x_final[train_index], df_x_final[test_index]
    train_y, test_y = df_y_230413[train_index], df_y_230413[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7810264900662252
np.mean(test_score1)         # 0.7653921568627451



# =============================================================================
# =============================================================================
# 결론 : 비율 포함한 모델은 좋지 않음
# =============================================================================
# 비율 그자체도 추가
df_x_230413.to_csv('승하차_일반인_컬럼작업완_230414.csv', encoding = 'cp949')
df_x_230413 = pd.read_csv('승하차_일반인_컬럼작업완_230414.csv', encoding = 'cp949', index_col=0)

# 비율 컬럼만 제외하고 나머진 로그변환
df_x_log1_230413 = np.log1p(df_x_230413)
df_x_log1_230413['출근_하차_일반인비율'] = type_out_start_result['일반인비율']
df_x_log1_230413['퇴근_승차_일반인비율'] = type_in_finish_result['일반인비율']

df_x_230413.hist()
train_x, test_x, train_y, test_y = train_test_split(df_x_log1_230413.values, df_y_230413, random_state=0)


# 7. SequentialFeatureSelector로 최적독립변수 조합 확인
from sklearn.linear_model import LogisticRegression as lr
from mlxtend.feature_selection import SequentialFeatureSelector as sFs
m_lr = lr()
sfs = sFs(m_lr, k_features='best', forward=False, verbose=2, scoring='accuracy', cv=4)
sfs.fit(df_x_log1_230413, df_y_230413)
dir(SequentialFeatureSelector)
# 8. 결과 출력
print("선택된 변수: ", sfs.k_feature_names_)
df_x_230413.columns[[1, 3, 4, 6, 7]]
print("검증 정확도: ", sfs.k_score_)
# 선택된 변수:  ('1', '3', '4', '6', '7')
# ['퇴근시간대', '스타벅스수', '출근시간대승차인원', '출근시간대하차인원', '퇴근시간대하차인원']
# 검증 정확도:  0.7756862745098039


df_x_sc = df_x_230413.loc[:, ['퇴근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '퇴근시간대일반인승차']]
from sklearn.preprocessing import MinMaxScaler as minmax
m_mm = minmax()
x_sc = m_mm.fit_transform(df_x_sc)
DataFrame(x_sc).hist()

train_x, test_x, train_y, test_y = train_test_split(x_sc, df_y_230413, random_state=0)
m_lr = lr()
result = m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7266666666666667
m_lr.score(test_x, test_y)             # 0.8235294117647058

from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(x_sc):
    train_x, test_x = x_sc[train_index], x_sc[test_index]
    train_y, test_y = df_y_230413[train_index], df_y_230413[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7594701986754968
np.mean(test_score1)         # 0.7208823529411765

# =============================================================================
# =============================================================================
df_x = df_x_230413.loc[:, ['출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원']]
df_x.hist()
df_x_log = np.log1p(df_x).values
train_x, test_x, train_y, test_y = train_test_split(df_x_log, df_y_230413, random_state=0)
m_lr = lr()
result = m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7533333333333333
m_lr.score(test_x, test_y)             # 0.8627450980392157

from sklearn.model_selection import KFold
m_cv1 = KFold(4)       # train : test 비율 약 75 : 25

train_score1 = []; test_score1 = []
for train_index, test_index in m_cv1.split(df_x_log):
    train_x, test_x = df_x_log[train_index], df_x_log[test_index]
    train_y, test_y = df_y_230413[train_index], df_y_230413[test_index]

    m_lr = lr()
    m_lr.fit(train_x, train_y)
    
    train_score1.append(m_lr.score(train_x, train_y))
    test_score1.append(m_lr.score(test_x, test_y))

# 교차검증 후 최종 점수
np.mean(train_score1)        # 0.7677593818984547
np.mean(test_score1)         # 0.7604901960784314


# =============================================================================
# =============================================================================
# 9. RandomForestClassifier 모델링
train_x, test_x, train_y, test_y = train_test_split(df_x_230413, df_y_230413, random_state=0)
df_x_log_123 = np.log1p(df_x_230413).values
v_columns = []; v_score = []
for i in range(10, 100):
    m_rfc = rf_c(n_estimators=i, max_depth = 5)
    sfs = sFs(m_rfc, k_features='best', forward=False, verbose=2, scoring='accuracy', cv=4)
    sfs.fit(df_x_230413, df_y_230413)
    vres = list(Series(sfs.k_feature_names_).map(lambda x : int(x)))
    v_columns.append(df_x_230413.columns[[vres]])
    v_score.append(sfs.k_score_)
    df_x_230413.columns[[0,1,2]]

# 11. 결과 출력
print("선택된 변수: ", sfs.k_feature_names_)
df_x_230413.columns[[0, 1, 2, 5, 6, 7]]
print("검증 정확도: ", sfs.k_score_)
# 선택된 변수:  ('0', '1', '2', '5', '6', '7')
# ['출근시간대', '퇴근시간대', '대합실면적', '퇴근시간대승차인원', '출근시간대하차인원', '퇴근시간대하차인원']
# 검증 정확도:  0.8001422475106686


m_rfc = rf_c()
m_rfc.fit(train_x, train_y)
m_rfc.score(train_x, train_y)     # 1.0
m_rfc.score(test_x, test_y)       # 0.7647058823529411
m_rfc.feature_importances_
# [0.17114952, 0.22007273, 0.28188328, 0.16889182, 0.15800265]
# ['출근시간대', '퇴근시간대', '대합실면적', '사업체수', '스타벅스수']

sfs = sFs(m_rfc, k_features='best', forward=True, verbose=2, scoring='accuracy', cv=5)
sfs.fit(df_x_230413.values, df_y_230413)

# 결과 출력
print("선택된 변수: ", sfs.k_feature_names_)
# ('0', '1', '2', '5', '6', '7', '8')
# ['출근시간대', '퇴근시간대', '대합실면적', '퇴근시간대승차인원', '출근시간대하차인원', '퇴근시간대하차인원',
# '출근시간대일반인하차']
print("검증 정확도: ", sfs.k_score_)
vres = list(Series(sfs.k_feature_names_).map(lambda x : int(x)))
sfs.k_feature_names_
list(df_x_230413.columns[vres])

# ['출근시간대', '퇴근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '퇴근시간대하차인원', '출근시간대일반인하차']
# ('0', '1', '2', '3', '5', '7', '8')

# 처음 start
# 일평균승하차, 인구수, 사업체수, 종사자수
# 일평균 => 출퇴근시간대/승차/하차/max
# 인구수 => qgis내 역 좌표값을 근거로 distance 반경 1km에 해당하는 인구데이터 출력
#           model을 만들었으나, 유의미하지 않았다.
# 동일한 방법으로 사업체수, 종사자수 반영 > grid에 따른 산출방식이 무의미하다고 확인
# ex) 서울 내 사업체수 164만개인데, 우리데이터는 어떠하다~
# 동별로 산출 => 상대적으로 유의미한 변수로 작용
# 다만, 로지스틱 회귀모델에 변수로 사용되진 않음
# qgis로 스타벅스 수 => distance 500m 내 점포개수 산출 -> 매우 유의미함
# 500m 로 선정한 이유 : 역세권 == 500m 이내
# '역세권 장기전세주택 건립관련 정비계획 수립 및 운영 기준' 제 3절 용어의 정의  참고
# 갓 스타벅스! / 이디야와 올리브영이 스타벅스 매장 위치를 참고하여 자신들의 점포 위치를 선정한다고 하는데
#               아주 매우 합리적인 선택임 ('허브 앤드 스포크'(hub and spoke) 전략)

# 최종 모델 반영 변수 : '출근시간대', '대합실면적', '스타벅스수', '퇴근시간대승차인원', '출근시간대하차인원'
# 변수 해석
# 0) 출근시간대 : 06~10시 (일반적으로 06~09시로 하나, 최근 10 to 19 트랜드 반영?)
# 1) 퇴근시간대 : 17~20시
# 2) 대합실면적 종속변수가 이러이러해서 면적이 큰 영향을 어쩌고 저쩌고
# 3) 스타벅스수
# 4) 퇴근시간대승차인원, 출근시간대하차인원 : 내용적으로 동일한 변수라고 생각됨
#    퇴근시간대에 승차하는 인원이 많다 = 업무지구일 가능성이 높음
#    출근시간대하차인원 상동

# 생각해볼점
# 퇴근시간대 vs 출근시간대 max 값
# -
# 변수들의 분포가 왜곡되어 있으면, 모형에서 잘 각 역별로 적게는 10단위, 크게는 2~300 단위의 차이를 보임
# - 출근시간대는 유의미한데, 퇴근시간대는 why?
# 연령대를 고려한 변수도 사용해 보았으나,
# 결론적으로 연령 상관없이 총 노출도(전체 인구)만 고려했다고 생각됨
# 기존 종속변수 = 4개 범주
# 3, 4등급에 해당하는 역들이 과연 광고 금액 및 광고효과(노출도 측면에서)가 다르다고 할 수 없어 보임
# why?
# 3진분류로 모형 예측시, train/test score 가 65점을 넘어가지 않음 (이런 판단이 합리적인가? 는 다른 문제)
# - 광고를 본다는 관점에서
#   승차를 위해 기다리면서는 광고에 노출된다고 생각함
#   하차하면서 광고를 본다고? 뒤돌아서?

# 최종 독립변수 조합의 경우,  변수 스케일링 전후 score 차이 거의 없음
# minmax, standard 모두 그닥
# 헌데, log 변환시 정확도 5점 정도 상승
# 로지스틱 회귀분석에서 변수의 편포를 정규분포에 가깝게 변환하는 것이 왜? 성능을 향상시키는가?
# feat by chatGPT 잘못된 가중치를 할당할 가능성이 있습니다.
# 예를 들어, 변수들이 왜곡되어 있으면 이상치(outlier)에 더 큰 가중치를 부여할 수 있습니다.
# 이 경우 모형은 이상치를 더 중요한 변수로 인식하게 되어 예측력이 떨어집니다.
# 따라서, 변수들의 분포를 정규분포에 가깝게 변환하면, 이상치의 영향력을 줄일 수 있습니다.

# 내 생각
# 스케일링 및 로그변환 전 boxplot을 그려보면
# 매우 많은 이상치가 표시됨
# 다만, 이상치들을 사분위수 변환을 통해
# if x > Q3 + 1.5IQR => return Q3 + 1.5IQR 로 변환 작업 실시 후,
# 모델에 fitting => 점수 떨어짐
# 분포의 왜곡을 완화 => 이상치에 덜 민감 => 로그변환의 타당함
# 또한, boxplot에서는 이상치로 취급하지만, 실제로는 유의미한 데이터일 가능성
# ex) 강남역 승차인원이 다른 역들에 비해 과하게 많다고 해서 유의미하지 않은 것은 아니라고 생각!








