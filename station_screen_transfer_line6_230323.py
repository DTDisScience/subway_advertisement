# -*- coding: utf-8 -*-
%run my_profile.py
from datetime import datetime
import locale
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')

# =============================================================================
# 6호선 편집
# =============================================================================
sta6_before = pd.read_csv('광고사업진행역_일별_승하차인원_6호선.csv')
screen6_adv = pd.read_csv('광고판개수_환승역개수_급지분류_6호선.csv')

# STEP 0) 호선 컬럼 삭제
sta6_before = sta6_before.drop('호선', axis=1)

# STEP 1) 휴일여부 컬럼 추가
def holiday_function(x, y) :
    holiday_list = ['2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02',
                    '2022-03-01', '2022-05-05', '2022-05-08', '2022-06-06',
                    '2022-08-15', '2022-09-09', '2022-09-10', '2022-09-11',
                    '2022-09-12', '2022-10-03', '2022-10-09', '2022-10-10',
                    '2022-12-25']
    holiday_list_1 = []
    for i in holiday_list :
        holiday_list_1.append(datetime.strptime(i, '%Y-%m-%d'))
    
    x = datetime.strptime(x, '%Y-%m-%d')
    
    if x in holiday_list_1 :
        return '주말'
    elif (y == '토') | (y == '일') :
        return '주말'
    else :
        return '평일'
    
sta6_before['휴일여부'] = list(map(holiday_function, sta6_before['수송일자'], sta6_before['요일']))

# STEP 2) 일평균, 영업일 평균, 비영업일 평균 승하차인원 계산
df2 = sta6_before.groupby('역명')[['승하차인원']].mean().reset_index()
df2_avg_business_day = sta6_before.loc[sta6_before['휴일여부'] == '평일', :].groupby('역명')[['승하차인원']].mean().reset_index()
df2_avg_non_business_days = sta6_before.loc[sta6_before['휴일여부'] == '주말', :].groupby('역명')[['승하차인원']].mean().reset_index()
df2['승하차인원'] = df2['승하차인원'].map(lambda x : round(x))
df2['영업일_평균_승하차'] = df2_avg_business_day['승하차인원'].map(lambda x : round(x))
df2['비영업일_평균_승하차'] = df2_avg_non_business_days['승하차인원'].map(lambda x : round(x))

# STEP 3) 각 데이터 float → int 타입 변환
df2 = df2.astype({'승하차인원':'int32', '영업일_평균_승하차':'int64', '비영업일_평균_승하차':'int64'})

# STEP 4) 역명 기준으로 오름차순(가나다순) 정렬
df2 = df2.sort_values('역명', ascending = True)
screen6_adv = screen6_adv.sort_values('역명', ascending = True)

# =============================================================================
# 각 데이터프레임 역명 & 순서 동일한지 확인
# =============================================================================
def name_function(x, y) :
    if x == y :
        return 'O'
    else :
        return x + '_' + y

a1 = list(map(name_function, df2['역명'], screen6_adv['역명']))
vres = []
for i in a1:
    if i != 'O':
        vres.append(i)

# vres가 빈 리스트이면 정확하게 일치함

# =============================================================================
# 환승노선수, 급지분류, 스크린수, 스크린_포스터수 컬럼 추가
# =============================================================================
df2['환승노선수'] = screen6_adv['환승개수']
df2['급지분류'] = screen6_adv['급지분류']
df2['스크린수'] = screen6_adv['스크린']
df2['스크린_포스터수'] = screen6_adv['스크린_포스터']
df2 = df2.iloc[:, [0,5,1,2,3,4,6,7]]
df2 = df2.rename(columns={'승하차인원':'일평균_승하차'})

# =============================================================================
# 월평균 승하차 컬럼 추가
# =============================================================================
sta6_before['월'] = sta6_before['수송일자'].map(lambda x : str(x)[5:7])
month_df2 = sta6_before.groupby(['월', '역명'])[['승하차인원']].sum().reset_index()
df2_month_avg = month_df2.groupby('역명')[['승하차인원']].mean().reset_index()

df2['월평균_승하차'] = df2_month_avg['승하차인원'].map(lambda x : round(x))
df2 = df2.iloc[: , [0,1,2,8,3,4,5,6,7]]



df1.loc[df1['역명'] == '신내',:]

# =============================================================================
# 스크린_포스터 컬럼 => 포스터 컬럼으로 변경
# =============================================================================
df1['포스터수'] = list(map(lambda x, y : x - y, df1['스크린_포스터수'], df1['스크린수']))
df1 = df1.drop(['스크린_포스터수', '포스터'], axis=1)

df2['포스터수'] = list(map(lambda x, y : x - y, df2['스크린_포스터수'], df2['스크린수']))
df2 = df2.drop(['스크린_포스터수'], axis=1)

df2.to_csv('6호선_컬럼작업중_230323.csv', index=False)
df2 = pd.read_csv('6호선_컬럼작업중_230323.csv')

# =============================================================================
# 면적깊이
# =============================================================================
dongwoo_df = pd.read_csv('면적깊이.csv')

dongwoo_df_1 = dongwoo_df.sort_values(['호선', '역명'], ascending = [True, True])
dongwoo_df_1

# STEP 1) 1호선
v1 = dongwoo_df_1.loc[dongwoo_df_1['호선'] == 1, '역명'].unique()
a1 = df1.loc[df1['호선'] == 1, '역명'].unique()

len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 1, :])   # 10 행
for i in v1:
    if i not in a1 :
        num1 = dongwoo_df_1[(dongwoo_df_1['호선'] == 1) & (dongwoo_df_1['역명'] == i)].index
        dongwoo_df_1.drop(num1, inplace=True)       
dongwoo_df_1.reset_index(drop=True, inplace=True)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 1, :])   # 10 행

# STEP 2) 2호선
# '까치산', '도림천', '동대문역사문화공원', '상왕십리', '신답', '신설동', '용답', '용두', '종합운동장'
v1 = dongwoo_df_1.loc[dongwoo_df_1['호선'] == 2, '역명'].unique()
a1 = df1.loc[df1['호선'] == 2, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 2, :])   # 51 행
for i in v1:
    if i not in a1 :
        num1 = dongwoo_df_1[(dongwoo_df_1['호선'] == 2) & (dongwoo_df_1['역명'] == i)].index
        dongwoo_df_1.drop(num1, inplace=True)       
dongwoo_df_1.reset_index(drop=True, inplace=True)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 2, :])   # 42 행
len(df1.loc[df1['호선'] == 2, :])                     # 42 행

# STEP 3) 3호선
# '가락시장', '경찰병원', '오금', '옥수', '지축'
v1 = dongwoo_df_1.loc[dongwoo_df_1['호선'] == 3, '역명'].unique()
a1 = df1.loc[df1['호선'] == 3, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 3, :])   # 34 행
for i in v1:
    if i not in a1 :
        num1 = dongwoo_df_1[(dongwoo_df_1['호선'] == 3) & (dongwoo_df_1['역명'] == i)].index
        dongwoo_df_1.drop(num1, inplace=True)       
dongwoo_df_1.reset_index(drop=True, inplace=True)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 3, :])   # 29 행
len(df1.loc[df1['호선'] == 3, :])                     # 29 행

# STEP 4) 4호선
# '남태령', '당고개', '동대문', '동작', '한성대입구'
v1 = dongwoo_df_1.loc[dongwoo_df_1['호선'] == 4, '역명'].unique()
a1 = df1.loc[df1['호선'] == 4, '역명'].unique()
dongwoo_df_1['역명'] = dongwoo_df_1['역명'].map(lambda x : '미아사거리' if x == '미아' else x)
dongwoo_df_1['역명'] = dongwoo_df_1['역명'].map(lambda x : '서울역' if x == '서울' else x)
dongwoo_df_1['역명'] = dongwoo_df_1['역명'].map(lambda x : '성신여대' if x == '성신여대입구' else x)
dongwoo_df_1['역명'] = dongwoo_df_1['역명'].map(lambda x : '총신대' if x == '총신대입구' else x)
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 4, :])   # 26 행
for i in v1:
    if i not in a1 :
        num1 = dongwoo_df_1[(dongwoo_df_1['호선'] == 4) & (dongwoo_df_1['역명'] == i)].index
        dongwoo_df_1.drop(num1, inplace=True)
dongwoo_df_1.reset_index(drop=True, inplace=True)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 4, :])   # 20 행
len(df1.loc[df1['호선'] == 4, :])                     # 20 행
dongwoo_df_1.drop(84, inplace=True)
dongwoo_df_1.reset_index(drop=True, inplace=True)


# STEP 5) 5호선
# '강일', '미사', '하남검단산', '하남시청', '하남풍산'
v1 = dongwoo_df_1.loc[dongwoo_df_1['호선'] == 5, '역명'].unique()
a1 = df1.loc[df1['호선'] == 5, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 5, :])   # 56 행
for i in v1:
    if i not in a1 :
        num1 = dongwoo_df_1[(dongwoo_df_1['호선'] == 5) & (dongwoo_df_1['역명'] == i)].index
        dongwoo_df_1.drop(num1, inplace=True)       
dongwoo_df_1.reset_index(drop=True, inplace=True)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 5, :])   # 51 행
len(df1.loc[df1['호선'] == 5, :])                     # 51 행

# =============================================================================
# 7호선 삭제 목록
# 없음
v1 = dongwoo_df_1.loc[dongwoo_df_1['호선'] == 7, '역명'].unique()
a1 = df1.loc[df1['호선'] == 7, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 7, :])   # 56 행
for i in v1:
    if i not in a1 :
        num1 = dongwoo_df_1[(dongwoo_df_1['호선'] == 7) & (dongwoo_df_1['역명'] == i)].index
        dongwoo_df_1.drop(num1, inplace=True)       
dongwoo_df_1.reset_index(drop=True, inplace=True)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 7, :])   # 51 행
len(df1.loc[df1['호선'] == 7, :])                     # 51 행


# =============================================================================
# 8호선 삭제 목록
# '남위례'
v1 = dongwoo_df_1.loc[dongwoo_df_1['호선'] == 8, '역명'].unique()
a1 = df1.loc[df1['호선'] == 8, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 8, :])   # 18 행
for i in v1:
    if i not in a1 :
        num1 = dongwoo_df_1[(dongwoo_df_1['호선'] == 8) & (dongwoo_df_1['역명'] == i)].index
        dongwoo_df_1.drop(num1, inplace=True)       
dongwoo_df_1.reset_index(drop=True, inplace=True)
len(dongwoo_df_1.loc[dongwoo_df_1['호선'] == 8, :])   # 17 행
len(df1.loc[df1['호선'] == 8, :])                     # 17 행

# =============================================================================
# 6호선 제외 파일 역명 일치 확인
# =============================================================================
def name_function(x, y) :
    if x == y :
        return 'O'
    else :
        return x + '_' + y

a1 = list(map(name_function, df1['역명'], dongwoo_df_2['역명']))
vres = []
for i in a1:
    if i != 'O':
        vres.append(i)

dongwoo_df_2 = dongwoo_df_1.loc[dongwoo_df_1['호선'] != 6, :]
dongwoo_df_2.reset_index(drop=True, inplace=True)


# =============================================================================
# 합치기
# =============================================================================
df1['대합실면적(m^2)'] = dongwoo_df_2['대합실면적(m^2)']
df1['승강장면적(m^2)'] = dongwoo_df_2['승강장면적(m^2)']
df1['형식'] = dongwoo_df_2['형식']
df1['층수'] = dongwoo_df_2['층수']
df1['정거장깊이(m)'] = dongwoo_df_2['정거장깊이(m)']

df1.to_csv('6호선제외_컬럼작업중_230323.csv', index=False)

# =============================================================================
# 6호선 파일 생성
# =============================================================================
# '신내' 삭제
dongwoo_df_6 = dongwoo_df_1.loc[dongwoo_df_1['호선'] == 6, :]
dongwoo_df_6.reset_index(drop=True, inplace=True)

v1 = dongwoo_df_6['역명'].unique()
a1 = df2['역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)

for i in v1:
    if i not in a1 :
        num1 = dongwoo_df_6[dongwoo_df_6['역명'] == i].index
        dongwoo_df_6.drop(num1, inplace=True)       
dongwoo_df_6.reset_index(drop=True, inplace=True)


# =============================================================================
# 6호선 파일 합치기
# =============================================================================
df2['대합실면적(m^2)'] = dongwoo_df_6['대합실면적(m^2)']
df2['승강장면적(m^2)'] = dongwoo_df_6['승강장면적(m^2)']
df2['형식'] = dongwoo_df_6['형식']
df2['층수'] = dongwoo_df_6['층수']
df2['정거장깊이(m)'] = dongwoo_df_6['정거장깊이(m)']

df2.to_csv('6호선_컬럼작업중_230323.csv', index=False)

# =============================================================================
# 은희씨 파일 합치기
# =============================================================================
hee_df_2 = pd.read_csv('hee_6호선.csv', encoding = 'cp949')

df2 = df2.drop(25)
df2.reset_index(drop=True, inplace=True)
def name_function(x, y) :
    if x == y :
        return 'O'
    else :
        return x + '_' + y

a1 = list(map(name_function, df2['역명'], hee_df_2['역명']))
vres = []
for i in a1:
    if i != 'O':
        vres.append(i)

df2['대학여부'] = hee_df_2['대학여부']
df2['포스터수'] = list(map(lambda x, y : x - y, df2['스크린_포스터수'], df2['스크린수']))
df2 = df2.drop('스크린_포스터수', axis=1)
df3 = df2.iloc[:, [0,1,2,3,4,5,6,7,14,8,9,10,11,12,13]]

df3.to_csv('6호선_컬럼작업중_230323.csv', index=False)

# =============================================================================
# 윤주씨 파일 합치기
# =============================================================================
joo_df_6 = joo_df.loc[joo_df['호선'] == 6, :]
joo_df_6.reset_index(drop=True, inplace=True)
joo_df_6 = joo_df_6.drop(26)
joo_df_6.reset_index(drop=True, inplace=True)
joo_df_6 = joo_df_6.drop(21)
joo_df_6.reset_index(drop=True, inplace=True)
joo_df_6
joo_df_6['상가여부'] = joo_df_6['상가여부'].astype('int')
joo_df_6['주민등록인구수']
joo_df_5 = pd.read_csv('joo_line6.csv')

joo_df_5 = joo_df_5.drop([36,38])
joo_df_5 = joo_df_5.sort_values('역명', ascending = True)
joo_df_5.reset_index(drop=True, inplace=True)

joo_df_5['주민등록인구수'] = joo_df_5['주민등록인구수'].astype('int')
joo_df_5['상가여부'] = joo_df_5['상가여부'].astype('int')


df3['광역시'] = joo_df_6['광역시']
df3['행정동'] = joo_df_6['행정동']
df3['주민등록인구수'] = joo_df_5['주민등록인구수']
df3['상가여부'] = joo_df_5['상가여부']

df3 = df3.iloc[:, [0,1,15,16,17,18,2,3,4,5,6,7,8,9,10,11,12,13,14]]
len(df3.columns)  # 19
df3.to_csv('6호선_컬럼작업중_230323.csv', index=False)
df1
 역명  급지분류    광역시  행정동  주민등록인구수  일평균_승하차  월평균_승하차  영업일_평균_승하차  
역명  급지분류    광역시    행정동  주민등록인구수  상가여부  일평균_승하차  월평균_승하차  영업일_평균_승하차
