# -*- coding: utf-8 -*-
%run my_profile.py
from datetime import datetime
import locale
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')

sta_before = pd.read_csv('광고사업진행역_일별_승하차인원_6호선제외.csv')
screen_adv = pd.read_csv('광고판개수_환승역개수_급지분류_6호선제외.csv')

# =============================================================================
# 주말(2022년 공휴일 및 대체공휴일 포함) / 평일 구분하기
# =============================================================================
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

sta_before['휴일여부'] = list(map(holiday_function, sta_before['수송일자'], sta_before['요일']))

df1 = sta_before.groupby(['호선', '역명'])[['승하차인원']].mean().reset_index()
df_avg_business_day = sta_before.loc[sta_before['휴일여부'] == '평일', :].groupby(['호선', '역명'])[['승하차인원']].mean().reset_index()
df_avg_non_business_days = sta_before.loc[sta_before['휴일여부'] == '주말', :].groupby(['호선', '역명'])[['승하차인원']].mean().reset_index()

df1['승하차인원'] = df1['승하차인원'].map(lambda x : round(x))
df1['영업일_평균_승하차'] = df_avg_business_day['승하차인원'].map(lambda x : round(x))
df1['비영업일_평균_승하차'] = df_avg_non_business_days['승하차인원'].map(lambda x : round(x))
df1 = df1.astype({'승하차인원':'int32', '영업일_평균_승하차':'int64', '비영업일_평균_승하차':'int64'})
df1.rename(columns={'승하차인원':'일평균_승하차'})
df1 = df1.sort_values(['호선', '역명'], ascending = [True, True])
df1.drop(1, inplace=True))
df1.reset_index(drop=True, inplace=True)

# =============================================================================
# 역명 통일 / 호선&역명별 오름차순 정렬
# =============================================================================
df1.to_csv('작업중_1.csv', index=False)
screen_adv = screen_adv.sort_values(['호선', '역명'], ascending = [True, True])
screen_adv.to_csv('작업중_2.csv', index=False)

# 다음은 승하차정보가 없는 역
# 2호선 : 신천 - 잠실새내
# 4호선 : 서울역, 성신여대, 총신대
df1.loc[df1['역명'] == '성신여대입구', :]
df1['역명'] = df1['역명'].map(lambda x : '성신여대' if x == '성신여대입구' else x)
df1['역명'] = df1['역명'].map(lambda x : '총신대' if x == '총신대입구' else x)
screen_adv['역명'] = screen_adv['역명'].map(lambda x : '잠실새내' if x == '신천' else x)
screen_adv.reset_index(drop=True, inplace=True)

# =============================================================================
# 각 데이터프레임 역명 & 순서 동일한지 확인
# =============================================================================
def name_function(x, y) :
    if x == y :
        return 'O'
    else :
        return x + '_' + y

a1 = list(map(name_function, df1['역명'], screen_adv['역명']))
vres = []
for i in a1:
    if i != 'O':
        vres.append(i)


# =============================================================================
# 환승노선수, 급지분류, 스크린수, 스크린_포스터수 컬럼 추가
# =============================================================================
df1['환승노선수'] = screen_adv['환승개수']
df1['급지분류'] = screen_adv['급지분류']
df1['스크린수'] = screen_adv['스크린']
df1['스크린_포스터수'] = screen_adv['스크린_포스터']
df1 = df1.iloc[:, [0,1,6,2,3,4,5,7,8]]
df1 = df1.rename(columns={'승하차인원':'일평균_승하차'})


# =============================================================================
# 월평균 승하차 컬럼 추가
# =============================================================================
sta_before['월'] = sta_before['수송일자'].map(lambda x : str(x)[5:7])
month_df1 = sta_before.groupby(['월', '호선', '역명'])[['승하차인원']].sum().reset_index()
df1_month_avg = month_df1.groupby(['호선', '역명'])[['승하차인원']].mean().reset_index()

df1['월평균_승하차'] = df1_month_avg['승하차인원'].map(lambda x : round(x))
df1 = df1.iloc[: , [0,1,2,3,9,4,5,6,7,8]]

df1.to_csv('6호선제외_컬럼작업중_230323.csv', index=False)
df1 = pd.read_csv('6호선제외_컬럼작업중_230323.csv')

# =============================================================================
# 은희씨 파일 합치기
# =============================================================================
hee_df_1 = pd.read_csv('hee_6호선제외.csv', encoding = 'cp949')
df1['대학여부'] = hee_df_1['대학여부']

df1.to_csv('6호선제외_컬럼작업중_230323.csv', index=False)


# =============================================================================
# 윤주씨 파일 합치기
# =============================================================================
joo_df = pd.read_csv('joo_전체.csv')
joo_df['호선'] = joo_df['호선'].map(lambda x : x[0])
joo_df['역명'] = joo_df['역명'].map(lambda x : x.split('(')[0])
joo_df = joo_df.drop(0)
joo_df.reset_index(drop=True, inplace=True)

joo_df = joo_df.sort_values(['호선', '역명'], ascending = [True, True])
joo_df['호선'] = joo_df['호선'].astype('int')

# STEP 1) 2호선
# '도림천', '동대문역사문화공원', '상왕십리', '신답', '신설동', '용답', '용두', '종합운동장'
v1 = joo_df.loc[joo_df['호선'] == 2, '역명'].unique()
a1 = df1.loc[df1['호선'] == 2, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(joo_df.loc[joo_df['호선'] == 2, :])   # 50 행
for i in v1:
    if i not in a1 :
        num1 = joo_df[(joo_df['호선'] == 2) & (joo_df['역명'] == i)].index
        joo_df.drop(num1, inplace=True)       
joo_df.reset_index(drop=True, inplace=True)
len(joo_df.loc[joo_df['호선'] == 2, :])         # 42 행
len(df1.loc[df1['호선'] == 2, :])               # 42 행

# STEP 2) 3호선
# '가락시장', '경찰병원', '오금', '옥수', '지축'
v1 = joo_df.loc[joo_df['호선'] == 3, '역명'].unique()
a1 = df1.loc[df1['호선'] == 3, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(joo_df.loc[joo_df['호선'] == 3, :])   # 34 행
for i in v1:
    if i not in a1 :
        num1 = joo_df[(joo_df['호선'] == 3) & (joo_df['역명'] == i)].index
        joo_df.drop(num1, inplace=True)       
joo_df.reset_index(drop=True, inplace=True)
len(joo_df.loc[joo_df['호선'] == 3, :])         # 29 행
len(df1.loc[df1['호선'] == 3, :])               # 29 행

# STEP 3) 4호선
# '남태령', '당고개', '동대문', '동작', '미아', '한성대입구'
joo_df['역명'] = joo_df['역명'].map(lambda x : x.replace(' ', ''))
joo_df['역명'] = joo_df['역명'].map(lambda x : '서울역' if x == '서울' else x)
joo_df['역명'] = joo_df['역명'].map(lambda x : '성신여대' if x == '성신여대입구' else x)
joo_df['역명'] = joo_df['역명'].map(lambda x : '총신대' if x == '총신대입구' else x)
v1 = joo_df.loc[joo_df['호선'] == 4, '역명'].unique()
a1 = df1.loc[df1['호선'] == 4, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(joo_df.loc[joo_df['호선'] == 4, :])   # 26 행
for i in v1:
    if i not in a1 :
        num1 = joo_df[(joo_df['호선'] == 4) & (joo_df['역명'] == i)].index
        joo_df.drop(num1, inplace=True)       
joo_df.reset_index(drop=True, inplace=True)
len(joo_df.loc[joo_df['호선'] == 4, :])         # 20 행
len(df1.loc[df1['호선'] == 4, :])               # 20 행

# STEP 4) 5호선
# '강일', '미사', '하남검단산', '하남시청', '하남풍산'
v1 = joo_df.loc[joo_df['호선'] == 5, '역명'].unique()
a1 = df1.loc[df1['호선'] == 5, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(joo_df.loc[joo_df['호선'] == 5, :])   # 56 행
for i in v1:
    if i not in a1 :
        num1 = joo_df[(joo_df['호선'] == 5) & (joo_df['역명'] == i)].index
        joo_df.drop(num1, inplace=True)       
joo_df.reset_index(drop=True, inplace=True)
len(joo_df.loc[joo_df['호선'] == 5, :])         # 51 행
len(df1.loc[df1['호선'] == 5, :])               # 51 행

# STEP 5) 7호선
# 없음
v1 = joo_df.loc[joo_df['호선'] == 7, '역명'].unique()
a1 = df1.loc[df1['호선'] == 7, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(joo_df.loc[joo_df['호선'] == 7, :])   # 56 행
for i in v1:
    if i not in a1 :
        num1 = joo_df[(joo_df['호선'] == 7) & (joo_df['역명'] == i)].index
        joo_df.drop(num1, inplace=True)       
joo_df.reset_index(drop=True, inplace=True)
len(joo_df.loc[joo_df['호선'] == 7, :])         # 51 행
len(df1.loc[df1['호선'] == 7, :])               # 51 행

# STEP 6) 8호선
# '남위례'
v1 = joo_df.loc[joo_df['호선'] == 8, '역명'].unique()
a1 = df1.loc[df1['호선'] == 8, '역명'].unique()
vres = []
for i in v1:
    if i not in a1 :
        vres.append(i)
len(joo_df.loc[joo_df['호선'] == 8, :])   # 18 행
for i in v1:
    if i not in a1 :
        num1 = joo_df[(joo_df['호선'] == 8) & (joo_df['역명'] == i)].index
        joo_df.drop(num1, inplace=True)       
joo_df.reset_index(drop=True, inplace=True)
len(joo_df.loc[joo_df['호선'] == 8, :])         # 17 행
len(df1.loc[df1['호선'] == 8, :])               # 17 행

# =============================================================================
# 각 데이터프레임 역명 & 순서 동일한지 확인
# =============================================================================
def name_function(x, y) :
    if x == y :
        return 'O'
    else :
        return x + '_' + y

a1 = list(map(name_function, df1['역명'], joo_df_1['역명']))
vres = []
for i in a1:
    if i != 'O':
        vres.append(i)

joo_df_1 = joo_df.loc[joo_df['호선'] != 6, :]
joo_df_1.reset_index(drop=True, inplace=True)
joo_df_1['상가여부'] = joo_df_1['상가여부'].astype('int')
joo_df_1['주민등록인구수'] = joo_df_1['주민등록인구수'].astype('int')
joo_df_1['주민등록인구수']
joo_df_1.loc[pd.isnull(joo_df_1['주민등록인구수']) == True, :]

df1['광역시'] = joo_df_1['광역시']
df1['행정동'] = joo_df_1['행정동']
df1['주민등록인구수'] = joo_df_1['주민등록인구수']
df1['상가여부'] = joo_df_1['상가여부']

joo_df_1.loc[joo_df_1['행정동'] == '세종로', '주민등록인구수'] = 6728

df1 = df1.iloc[:, [0,1,2,16,17,18,3,4,5,6,7,8,9,10,11,12,13,14,15,19]]
len(df1.columns)  # 20
df1.columns

df1.to_csv('6호선제외_컬럼작업중_230323.csv', index=False)
