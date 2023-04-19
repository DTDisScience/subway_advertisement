# -*- coding: utf-8 -*-
%run my_profile.py

# =============================================================================
# minmax scaling 진행
# =============================================================================
no_6 = pd.read_csv('6호선제외_컬럼작업중_230323.csv')
in_6 = pd.read_csv('6호선_컬럼작업중_230323.csv')

# 1) 불필요 컬럼 삭제
st_no_6 = no_6.drop(['호선', '역명', '급지분류', '광역시','행정동', '월평균_승하차',
                     '영업일_평균_승하차', '비영업일_평균_승하차', '환승노선수', '대합실면적(m^2)',
                     '승강장면적(m^2)', '형식', '층수', '대학여부', '상가여부'], axis=1)
st_in_6 = in_6.drop(['역명', '급지분류', '광역시','행정동', '월평균_승하차',
                     '영업일_평균_승하차', '비영업일_평균_승하차', '환승노선수', '대합실면적(m^2)',
                     '승강장면적(m^2)', '형식', '층수', '대학여부', '상가여부'], axis=1)


# =============================================================================
# 1~4호선 스크린도어 m^2 당 평균단가
# =============================================================================
# P 등급 : 880,346 만 원
4000000/(332*147/10000)     # 819,604 만 원
3500000/(253*147/10000)     # 941,087 만 원
((4000000/(332*147/10000)) + (3500000/(253*147/10000)))/2   # 880,346 만 원

# SSA 등급 : 761,900 만 원
3500000/(332*147/10000)     # 717,154 만 원
3000000/(253*147/10000)     # 806,646 만 원
((3500000/(332*147/10000))+(3000000/(253*147/10000)))/2     # 761,900 만 원

# SA 등급 : 643,454 만 원
3000000/(332*147/10000)     # 614,703 만 원
2500000/(253*147/10000)     # 672,205 만 원
((3000000/(332*147/10000))+(2500000/(253*147/10000)))/2     # 643,454 만 원

# A 등급 : 446,895 만 원
2000000/(332*147/10000)     # 409,802 만 원
1800000/(253*147/10000)     # 483,988 만 원
((2000000/(332*147/10000))+(1800000/(253*147/10000)))/2     # 446,895 만 원


# =============================================================================
# 578호선 스크린도어 m^2 당 평균단가
# =============================================================================
# P 등급 : 683,750 만 원
3500000/(332*167/10000) 
3000000/(244*167/10000)
((3500000/(332*167/10000))+(3000000/(244*167/10000)))/2     # 683,750 만 원

# SSA 등급 : 577,306 만 원
3000000/(332*167/10000)
2500000/(244*167/10000)
((3000000/(332*167/10000))+(2500000/(244*167/10000)))/2     # 577,306 만 원

# SA 등급 : 470,863 만 원
2500000/(332*167/10000)
2000000/(244*167/10000)
((2500000/(332*167/10000))+(2000000/(244*167/10000)))/2     # 470,863 만 원

# A 등급 : 346,384 만 원
1800000/(332*167/10000)
1500000/(244*167/10000)
((1800000/(332*167/10000))+(1500000/(244*167/10000)))/2     # 346,384 만 원

# =============================================================================
# 신규작업 start 
# =============================================================================
station_df_1 = pd.read_csv('subway1_8.csv')
station_df_2 = station_df_1.drop(['호선', '역명', '급지분류', '광역시','행정동',
                                  '환승노선수', '대합실면적(m^2)','승강장면적(m^2)', '형식', '층수'],axis = 1)
station_df_2 = station_df_2.drop('월평균_승하차', axis=1)
station_df_2 = station_df_2.drop('포스터수', axis = 1)
station_df_2 = station_df_2.drop(['대학여부', '상가여부'], axis=1)

# =============================================================================
# 등급(종속변수) null 확인
# =============================================================================
station_df_1.loc[pd.isnull(station_df_1['등급']) == True, ['호선', '역명']]
# 78    3        학여울
# 82    4  동대문역사문화공원
# 102   5        개화산
# 117   5         마장
# 118   5         마천
# 124   5         방화
# 138   5         오금
# 183   7         장암

# =============================================================================
# 컬럼명 변경
# =============================================================================
station_df_2.columns = list(map(lambda x : x.replace('_', ''), Series(station_df_2.columns)))
station_df_2.columns = [i.split('(')[0] for i in station_df_2.columns]

# =============================================================================
# 등급 null 인 행 삭제
# =============================================================================
station_df_3 = station_df_2.drop([78,82,102,117,118,124,138,183])
station_df_3.reset_index(drop=True, inplace=True)

# =============================================================================
# x, y 분리
# =============================================================================
station_df_3_x = station_df_3.iloc[:, :6]
station_df_3_y = station_df_3['등급'].astype('int').astype('category')
station_y = np.array(station_df_3_y)

# =============================================================================
# 변수 스케일링
# =============================================================================
from sklearn.preprocessing import StandardScaler as standard
from sklearn.preprocessing import MinMaxScaler as minmax
# 1) model 생성
m_st = standard()
m_mm = minmax()

# 2) 훈련 (학습 / fitting)
station_st_x = m_st.fit_transform(station_df_3_x)
station_mm_x = m_mm.fit_transform(station_df_3_x)

# =============================================================================
# train/test 분리
# =============================================================================
from sklearn.model_selection import train_test_split
st_tr_x, st_te_x, st_tr_y, st_te_y = train_test_split(station_st_x, station_y, random_state=0)
mm_tr_x, mm_te_x, mm_tr_y, mm_te_y = train_test_split(station_mm_x, station_y, random_state=0)

# =============================================================================
# standard 모델 생성
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st_tr_x, st_tr_y)
m_lr.score(st_tr_x, st_tr_y)          # 56.95
m_lr.score(st_te_x, st_te_y)          # 56.86

# 2-1) 계수 확인
m_lr.coef_
# [[-0.57252538, -0.21615675, -0.50023027, -0.75379686, -1.130624,
#   -1.12824522,  0.14127167, -0.55552547, -0.00649696, -0.58728085,
#   -0.10999151, -0.07880981, -0.08312097]]
m_lr.intercept_     # 1.30830561
m_lr.decision_function(st_train_x.iloc[:5, :])
# 2-2)회귀 식
# 1.30830561 - 0.57252538*X1 -0.21615675*X2 - 0.50023027*X3 - 0.75379686*X4 ...

# 3. 예측
new_data = station_st_x[0:1,:]
m_lr.predict(new_data)            # cut off에 해석된 Y 예상값
m_lr.predict_proba(new_data)      # 실제 확률 / 1or2급지 : 0.04463863, 3급지 : 0.95536137 
m_lr.predict_proba(new_data).round(4)  # [0.    , 0.1971, 0.218 , 0.585 ]

# =============================================================================
# minmax 모델
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(mm_tr_x, mm_tr_y)
m_lr.score(mm_tr_x, mm_tr_y)         # 50.99
m_lr.score(mm_te_x, mm_te_y)         # 54.90


# =============================================================================
# 주민등록인구수, 일평균승하차, 스크린+포스터수, 정거장깊이 컬럼만 사용
# =============================================================================
station_df_3.columns = list(map(lambda x : x.replace('_', ''), Series(station_df_3.columns)))
station_df_3.columns = [i.split('(')[0] for i in station_df_3.columns]
station_df_3['조명포스터'] = list(map(lambda x, y : x + y, station_df_3['스크린수'], station_df_3['포스터수']))
station_df_4 = station_df_3.drop(['영업일평균승하차', '비영업일평균승하차', '스크린수', '포스터수'], axis=1)

# x, y 분리
st_x = station_df_4.iloc[:, [0,1,2,4]]
st_y = station_df_4['등급'].astype('int').astype('category')

# scaling
from sklearn.preprocessing import StandardScaler as standard
from sklearn.preprocessing import MinMaxScaler as minmax
# 1) model 생성
m_st = standard()
m_mm = minmax()

# 2) 훈련 (학습 / fitting)
st_st_x = m_st.fit_transform(st_x)
st_mm_x = m_mm.fit_transform(st_x)

# 3) train/test 분리
from sklearn.model_selection import train_test_split
st1_tr_x, st1_te_x, st1_tr_y, st1_te_y = train_test_split(st_st_x, st_y, random_state=0)
mm1_tr_x, mm1_te_x, mm1_tr_y, mm1_te_y = train_test_split(st_mm_x, st_y, random_state=0)

# 4) standard 로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st1_tr_x, st1_tr_y)
m_lr.score(st1_tr_x, st1_tr_y)         # 57.90
m_lr.score(st1_te_x, st1_te_y)         # 54.90

# 5) minmax 로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(mm1_tr_x, mm1_tr_y)
m_lr.score(mm1_tr_x, mm1_tr_y)         # 51.65
m_lr.score(mm1_te_x, mm1_te_y)         # 54.90

# 6) 스케일링 하지 않고 로지스틱 회귀 모델 생성
st2_tr_x, st2_te_x, st2_tr_y, st2_te_y = train_test_split(st_x, st_y, random_state=0)
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st2_tr_x, st2_tr_y)
m_lr.score(st2_tr_x, st2_tr_y)         # 49.00
m_lr.score(st2_te_x, st2_te_y)         # 39.21

# =============================================================================
# 2진분류 (1,2등급 : 0, 3,4등급 : 1)
# =============================================================================
# x, y 분리
st_x = station_df_4.iloc[:, [0,1,2,4]]
st_y = station_df_4['등급'].map(lambda x : 1 if x > 2 else 0).astype('category')

# scaling
from sklearn.preprocessing import StandardScaler as standard
from sklearn.preprocessing import MinMaxScaler as minmax
# 1) model 생성
m_st = standard()
m_mm = minmax()

# 2) 훈련 (학습 / fitting)
st_st_x = m_st.fit_transform(st_x)
st_mm_x = m_mm.fit_transform(st_x)

# 3) train/test 분리
from sklearn.model_selection import train_test_split
st3_tr_x, st3_te_x, st3_tr_y, st3_te_y = train_test_split(st_st_x, st_y, random_state=0)
mm3_tr_x, mm3_te_x, mm3_tr_y, mm3_te_y = train_test_split(st_mm_x, st_y, random_state=0)

# 4) standard 로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st3_tr_x, st3_tr_y)
m_lr.score(st3_tr_x, st3_tr_y)         # 72.84
m_lr.score(st3_te_x, st3_te_y)         # 74.50

# 5) minmax 로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(mm3_tr_x, mm3_tr_y)
m_lr.score(mm3_tr_x, mm3_tr_y)         # 68.87
m_lr.score(mm3_te_x, mm3_te_y)         # 68.62


# =============================================================================
# 3진분류 (1등급 : 0, 2,3등급 : 1, 4등급 : 2)
# =============================================================================
# x, y 분리
st_x = station_df_4.iloc[:, [0,1,2,4]]
st_y = station_df_4['등급'].map(gr_function).astype('category')
def gr_function(x):
    if x == 1 :
        return 0
    elif x == 4 :
        return 2
    else : 
        return 1


# scaling
from sklearn.preprocessing import StandardScaler as standard
from sklearn.preprocessing import MinMaxScaler as minmax
# 1) model 생성
m_st = standard()
m_mm = minmax()

# 2) 훈련 (학습 / fitting)
st_st_x = m_st.fit_transform(st_x)
st_mm_x = m_mm.fit_transform(st_x)

# 3) train/test 분리
from sklearn.model_selection import train_test_split
st4_tr_x, st4_te_x, st4_tr_y, st4_te_y = train_test_split(st_st_x, st_y, random_state=0)
mm4_tr_x, mm4_te_x, mm4_tr_y, mm4_te_y = train_test_split(st_mm_x, st_y, random_state=0)

# 4) standard 로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st4_tr_x, st4_tr_y)
m_lr.score(st4_tr_x, st4_tr_y)         # 66.88
m_lr.score(st4_te_x, st4_te_y)         # 66.66

# 5) minmax 로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(mm4_tr_x, mm4_tr_y)
m_lr.score(mm4_tr_x, mm4_tr_y)         # 65.56
m_lr.score(mm4_te_x, mm4_te_y)         # 76.47

# =============================================================================
# 3진분류 (1,2등급 : 0, 3등급 : 1, 4등급 : 2)
# =============================================================================
# x, y 분리
st_x = station_df_4.iloc[:, [0,1,2,4]]
st_y = station_df_4['등급'].map(gr_function).astype('category')
def gr_function(x):
    if x == 3 :
        return 1
    elif x == 4 :
        return 2
    else : 
        return 0


# scaling
from sklearn.preprocessing import StandardScaler as standard
from sklearn.preprocessing import MinMaxScaler as minmax
# 1) model 생성
m_st = standard()
m_mm = minmax()

# 2) 훈련 (학습 / fitting)
st_st_x = m_st.fit_transform(st_x)
st_mm_x = m_mm.fit_transform(st_x)

# 3) train/test 분리
from sklearn.model_selection import train_test_split
st5_tr_x, st5_te_x, st5_tr_y, st5_te_y = train_test_split(st_st_x, st_y, random_state=0)
mm5_tr_x, mm5_te_x, mm5_tr_y, mm5_te_y = train_test_split(st_mm_x, st_y, random_state=0)

# 4) standard 로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st5_tr_x, st5_tr_y)
m_lr.score(st5_tr_x, st5_tr_y)         # 58.27
m_lr.score(st5_te_x, st5_te_y)         # 58.82

# 5) minmax 로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(mm5_tr_x, mm5_tr_y)
m_lr.score(mm5_tr_x, mm5_tr_y)         # 54.96
m_lr.score(mm5_te_x, mm5_te_y)         # 56.86


# =============================================================================
# 각 등급별 개수
# =============================================================================
station_df_4.loc[station_df_4['등급'] == 1, :]   # 3개
station_df_4.loc[station_df_4['등급'] == 2, :]   # 66개
station_df_4.loc[station_df_4['등급'] == 3, :]   # 56개
station_df_4.loc[station_df_4['등급'] == 4, :]   # 77개

# =============================================================================
# 오버샘플링
# =============================================================================
from imblearn.over_sampling import BorderlineSMOTE
st_x = station_df_4.iloc[:, [0,1,2,4]]
st_y = station_df_4['등급'].map(lambda x : 1 if x > 2 else 0).astype('category')

m_bsmt = BorderlineSMOTE(random_state=0)
X_resample1, y_resample1 = m_bsmt.fit_resample(st_x, st_y)
Series(y_resample1).value_counts()

m_st = standard()
m_mm = minmax()
st_st_x = m_st.fit_transform(X_resample1)
st_mm_x = m_mm.fit_transform(X_resample1)

from sklearn.model_selection import train_test_split
st6_tr_x, st6_te_x, st6_tr_y, st6_te_y = train_test_split(st_st_x, y_resample1, random_state=0)
mm6_tr_x, mm6_te_x, mm6_tr_y, mm6_te_y = train_test_split(st_mm_x, y_resample1, random_state=0)

from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st6_tr_x, st6_tr_y)
m_lr.score(st6_tr_x, st6_tr_y)         # 69.84
m_lr.score(st6_te_x, st6_te_y)         # 64.17

from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(mm6_tr_x, mm6_tr_y)
m_lr.score(mm6_tr_x, mm6_tr_y)         # 70.85
m_lr.score(mm6_te_x, mm6_te_y)         # 68.65


# =============================================================================
# 
# =============================================================================
st_x = station_df_4.iloc[:, [0,1]]
st_y = station_df_4['등급'].map(lambda x : 1 if x > 2 else 0).astype('category')
m_st = standard()
m_mm = minmax()
st_st_x = m_st.fit_transform(st_x)
st_mm_x = m_mm.fit_transform(st_x)

from sklearn.model_selection import train_test_split
st7_tr_x, st7_te_x, st7_tr_y, st7_te_y = train_test_split(st_st_x, st_y, random_state=0)
mm7_tr_x, mm7_te_x, mm7_tr_y, mm7_te_y = train_test_split(st_mm_x, st_y, random_state=0)

from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st7_tr_x, st7_tr_y)
m_lr.score(st7_tr_x, st7_tr_y)         # 73.50
m_lr.score(st7_te_x, st7_te_y)         # 66.66

from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(mm7_tr_x, mm7_tr_y)
m_lr.score(mm7_tr_x, mm7_tr_y)         # 68.21
m_lr.score(mm7_te_x, mm7_te_y)         # 66.66

# =============================================================================
# 
# =============================================================================
# 1호선 2
(3500000+3000000+3000000+2500000)/4   # 3000000
# 1호선 3
np.mean([2500000, 2000000, 1800000])  # 2100000

# 2호선 1
np.mean([7000000, 5000000, 4000000])  # 5333333
# 2호선 2
np.mean([5000000, 4000000, 3000000])  # 4000000
# 2호선 3
np.mean([4000000, 3500000, 2500000])  # 3333333
# 2호선 4
np.mean([3000000, 200000])  # 1600000


# 3호선 2
np.mean([6000000, 5000000, 4000000])  # 5000000
# 3호선 3
np.mean([5000000, 4000000])  # 4500000
# 3호선 4
np.mean([3000000])  # 3000000


# 4호선 2
np.mean([6000000, 2500000, 4000000])  # 4166666
# 4호선 3
np.mean([5000000, 3500000, 2000000])  # 3500000
# 4호선 4
np.mean([3000000, 1500000])  # 2250000


# 5호선 2
np.mean([3000000, 2500000])  # 2750000
# 5호선 3
np.mean([2500000, 2000000])  # 2250000
# 5호선 4
np.mean([1800000, 1500000])  # 1650000

def price_function(x, y) :
    if x == 1 :
        if y == 2 :
            return 3000000
        else :
            return 2100000
    elif x == 2:
        if y == 1:
            return 5333333
        elif y == 2:
            return 4000000
        elif y == 3:
            return 3333333
        else : 
            return 1600000
    elif x == 3:
        if y == 2:
            return 5000000
        elif y == 3 :
            return 4500000
        else :
            return 3000000
    elif x == 4:
        if y == 2:
            return 4166666
        elif y == 3:
            return 3500000
        else :
            return 2250000
    else :
        if y == 2:
            return 2750000
        elif y == 3:
            return 2250000
        else :
            return 1650000

# =============================================================================
# 
# =============================================================================
station_df_1
st_x = station_df_1.drop(['급지분류', '광역시','행정동',
                                  '환승노선수', '대합실면적(m^2)','승강장면적(m^2)', '형식', '층수'],axis = 1)

st_x_1 = st_x.drop(['월평균_승하차', '영업일_평균_승하차', '비영업일_평균_승하차', '대학여부', '상가여부'], axis=1)

st_x_1['조명포스터'] = list(map(lambda x, y : x+y, st_x_1['스크린수'], st_x_1['포스터수']))
st_x_1['금액'] = list(map(price_function, st_x_1['호선'], st_x_1['등급']))

st_x_2 = st_x_1.drop(['호선', '역명', '스크린수', '포스터수'], axis=1)
st_x_3 = st_x_2.drop([78,82,102,117,118,124,138,183])
st_x_3.reset_index(drop=True, inplace=True)
st_x_4 = st_x_3.drop('등급', axis=1)
st_x_4.columns = Series(st_x_4.columns).map(lambda x : x.split('(')[0]).map(lambda x : x.replace('_',''))

a1 = list(st_x_4['금액'].unique())
b1 = sorted(a1)
vres = []
for i in b1 :
    vres.append(sum(st_x_4['금액'] == i))

