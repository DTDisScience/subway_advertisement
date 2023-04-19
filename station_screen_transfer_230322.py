# -*- coding: utf-8 -*-
%run my_profile.py
import clipboard
from datetime import datetime

# 역별 시간대별 승하차 정보 불러오기
df1 = pd.read_csv('서울교통공사_20221231.csv', encoding = 'cp949')
df2 = df1.groupby(['호선', '역명'])[['연번']].min()
df2 = df2.reset_index()
df2 = df2.drop('연번', axis=1)
df2['역명'] = df2['역명'].map(lambda x : x.split('(')[0])
df2.loc[df2['호선'] == 1, '역명']   # 1호선 10개
# =============================================================================
# 클립보드로 DataFrame 생성
station_pcd = pd.read_clipboard()
station_pcd_6 = pd.read_clipboard()
station_pcd = station_pcd.rename(columns={'전체':'스크린'})
station_pcd = station_pcd.rename(columns={'전체.1':'스크린_포스터'})
station_pcd_6 = station_pcd_6.rename(columns={'전체':'스크린'})
station_pcd_6 = station_pcd_6.rename(columns={'전체.1':'스크린_포스터'})

station_pcd = station_pcd.iloc[:, [0,1,2,3,11,12]]
station_pcd_6 = station_pcd_6.iloc[:, [0,1,2,3,11,12]]

# 광고판개수_환승역개수_급지분류.csv 파일 만들기
station_pcd.to_csv('광고판개수_환승역개수_급지분류_6호선제외.csv', index=False)
station_pcd_6.to_csv('광고판개수_환승역개수_급지분류_6호선.csv', index=False)

# count_station = df2.groupby('호선')[['역명']].count().drop(6)
# count_station['스크린_포스터'] = station_pcd.groupby('호선')['역명'].count()
# count_station
# =============================================================================

df1 = df1.drop(['연번', '고유역번호(외부역코드)'], axis=1)
df1 = df1.rename(columns={'06시이전' : '05-06시간대'})
a1 = list(df1.columns[:4])
b1 = list(df1.columns[4:].str[:2])
for i in b1:
    a1.append(i)
df1.columns = a1
df1['역명'] = df1['역명'].map(lambda x : x.split('(')[0])
df1 = df1.fillna(0)


# 승하차 구분 필요 없어 보임 => 승하차 합계로 변경 
df2 = df1.groupby(['수송일자', '호선', '역명']).sum().reset_index()

# 요일 컬럼추가
import locale
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
df2['수송일자'] = df2['수송일자'].map(lambda x : datetime.strptime(x, '%Y-%m-%d'))
df2['요일'] = df2['수송일자'].map(lambda x : x.strftime('%A')[0])

# 일별 합계
df2 = df2.set_index(['수송일자', '요일', '호선', '역명'])
df3 = df2.stack()
df3 = df3.reset_index()
df3.columns = ['수송일자', '요일', '호선', '역명', '시간대', '승하차인원']
df4 = df3.groupby(['수송일자', '요일', '호선', '역명'])[['승하차인원']].sum().reset_index()

# 6호선 파일 나누기
df5 = df4.loc[df4['호선'] != 6, :]
df5.to_csv('호선별_역별_일별_승하차인원_6호선제외.csv', index=False)
df6 = df4.loc[df4['호선'] == 6, :]
df6.to_csv('호선별_역별_일별_승하차인원_6호선.csv', index=False)
df6.loc[df6['역명'] == '연신내', :]
# =============================================================================
# 호선별 각 역명 삭제사유
# =============================================================================
# 서울교통공사_역별 일별 시간대별 승하차인원 정보 에 포함된 역 중
# 광고사업 진행하지 않는 역을 제외한다
# 또한, 환승역의 경우 특정 호선에서만 해당역을 관리하는 경우 해당 호선의 역만 남긴다.

# 1호선 동묘앞역 삭제 / 사유 : 6호선에서 관리
df5 = df5.loc[df5['역명'] != '동묘앞', :]

# 2호선 삭제 목록
# '도림천', '동대문역사문화공원', '상왕십리', '신답', '신설동', '용답', '용두', '잠실새내', '종합운동장'
# 2호선 데이터 3285개 삭제
v1 = df5.loc[df5['호선'] == 2, '역명'].unique()
a1 = station_pcd.loc[station_pcd['호선'] == 2, '역명'].unique()
b1 = sorted(a1)
vres = []
for i in v1:
    if i not in b1 :
        vres.append(i)

df5.loc[df5['호선'] == 2, :]   # 18250 행
for i in v1:
    if i not in b1 :
        num1 = df5[(df5['호선'] == 2) & (df5['역명'] == i)].index
        df5.drop(num1, inplace=True)       
df5.reset_index(drop=True, inplace=True)
df5.loc[df5['호선'] == 2, :]   # 14965 행

# =============================================================================
# 3호선 삭제 목록
# '가락시장', '경찰병원', '오금', '옥수', '지축'
# 3호선 데이터 1825개 삭제
v1 = df5.loc[df5['호선'] == 3, '역명'].unique()
a1 = station_pcd.loc[station_pcd['호선'] == 3, '역명'].unique()
b1 = sorted(a1)
vres = []
for i in v1:
    if i not in b1 :
        vres.append(i)
df5.loc[df5['호선'] == 3, :]   # 12207 행
for i in v1:
    if i not in b1 :
        num1 = df5[(df5['호선'] == 3) & (df5['역명'] == i)].index
        df5.drop(num1, inplace=True)       
df5.reset_index(drop=True, inplace=True)
df5.loc[df5['호선'] == 3, :]   # 10382 행

# =============================================================================
# 4호선 삭제 목록
# '남태령', '당고개', '동대문', '동작', '미아', '서울역', '성신여대입구', '총신대입구', '한성대입구'
# 4호선 데이터 3285개 삭제
v1 = df5.loc[df5['호선'] == 4, '역명'].unique()
a1 = station_pcd.loc[station_pcd['호선'] == 4, '역명'].unique()
b1 = sorted(a1)
vres = []
for i in v1:
    if i not in b1 :
        vres.append(i)
df5.loc[df5['호선'] == 4, :]   # 9490 행
for i in v1:
    if i not in b1 :
        num1 = df5[(df5['호선'] == 4) & (df5['역명'] == i)].index
        df5.drop(num1, inplace=True)       
df5.reset_index(drop=True, inplace=True)
df5.loc[df5['호선'] == 4, :]   # 6205 행


# =============================================================================
# 5호선 삭제 목록
# '강일', '미사', '하남검단산', '하남시청', '하남풍산'
# 5호선 데이터 1825개 삭제
v1 = df5.loc[df5['호선'] == 5, '역명'].unique()
a1 = station_pcd.loc[station_pcd['호선'] == 5, '역명'].unique()
b1 = sorted(a1)
vres = []
for i in v1:
    if i not in b1 :
        vres.append(i)
df5.loc[df5['호선'] == 5, :]   # 20440 행
for i in v1:
    if i not in b1 :
        num1 = df5[(df5['호선'] == 5) & (df5['역명'] == i)].index
        df5.drop(num1, inplace=True)       
df5.reset_index(drop=True, inplace=True)
df5.loc[df5['호선'] == 5, :]   # 18615 행

# =============================================================================
# 7호선 삭제 목록
# '상동', '부평구청', '까치울', '부천시청'
# 상동역 : 2022년 1월 1일 : 인천교통공사로 운영권 이관
# 7호선 데이터 9개 삭제
v1 = df5.loc[df5['호선'] == 7, '역명'].unique()
a1 = station_pcd.loc[station_pcd['호선'] == 7, '역명'].unique()
b1 = sorted(a1)
vres = []
for i in v1:
    if i not in b1 :
        vres.append(i)
df5.loc[df5['호선'] == 7, :]   # 15339 행
for i in v1:
    if i not in b1 :
        num1 = df5[(df5['호선'] == 7) & (df5['역명'] == i)].index
        df5.drop(num1, inplace=True)       
df5.reset_index(drop=True, inplace=True)
df5.loc[df5['호선'] == 7, :]   # 15330 행

# =============================================================================
# 8호선 삭제 목록
# '남위례'
# 8호선 데이터 365개 삭제
v1 = df5.loc[df5['호선'] == 8, '역명'].unique()
a1 = station_pcd.loc[station_pcd['호선'] == 8, '역명'].unique()
b1 = sorted(a1)
vres = []
for i in v1:
    if i not in b1 :
        vres.append(i)
df5.loc[df5['호선'] == 8, :]   # 6570 행
for i in v1:
    if i not in b1 :
        num1 = df5[(df5['호선'] == 8) & (df5['역명'] == i)].index
        df5.drop(num1, inplace=True)       
df5.reset_index(drop=True, inplace=True)
df5.loc[df5['호선'] == 8, :]   # 6205 행

df5.to_csv('광고사업진행역_일별_승하차인원_6호선제외.csv', index=False)

# =============================================================================
# 6호선 삭제 목록
# '신내'
# 6호선 데이터 25개 삭제
v1 = df6['역명'].unique()
a1 = station_pcd_6['역명'].unique()
b1 = sorted(a1)
vres = []
for i in v1:
    if i not in b1 :
        vres.append(i)
df6.loc[df6['호선'] == 6, :]   # 13594 행
for i in v1:
    if i not in b1 :
        num1 = df6[df6['역명'] == i].index
        df6.drop(num1, inplace=True)       
df6.reset_index(drop=True, inplace=True)
df6.loc[df6['호선'] == 6, :]   # 13569 행

df6.to_csv('광고사업진행역_일별_승하차인원_6호선.csv', index=False)


# =============================================================================
# 일별 승하차인원 선별 완료 + 광고사업 진행하는역 분류 완료
# =============================================================================
# =============================================================================
# 주말 / 평일 기준 승하차인원 분류
# =============================================================================
df5 = df5.drop('수송일자',axis=1)
df5.groupby(['호선', '요일', '역명'])[['승하차인원']].agg(['sum', 'mean']).reset_index()

df6 = df5.rename(columns={'요일':'평일_주말'})
def f1(x) :
    if x in ['토', '일'] :
        return '주말'
    else :
        return '평일'

df6['평일_주말'] = df6['평일_주말'].map(f1)
df6.groupby(['호선', '평일_주말', '역명'])[['승하차인원']].agg(['sum', 'mean']).reset_index()





