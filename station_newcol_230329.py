# -*- coding: utf-8 -*-
%run my_profile.py


# =============================================================================
# 
# =============================================================================
df1 = pd.read_csv('6호선제외_컬럼작업중_230323.csv', encoding = 'cp949')
df1.to_csv('6호선제외_컬럼작업중_230323.csv', encoding = 'cp949')
df1 = df1.drop(77)
df1
df2 = df1.iloc[:, :2]

station_pcd = pd.read_clipboard()
station_pcd_1 = station_pcd.drop(['환승개수', '급지분류', '이상', '이하'], axis=1)
station_pcd_2 = station_pcd_1.sort_values(['호선', '역명'], ascending = [True, True])

df3 = pd.concat([df2, station_pcd_2.iloc[:, 2:]], axis=1)
df3

df1['일평균_승하차']
# 2) boxplot
plt.figure()
plt.boxplot(station_df_3_x)
df1['일평균_승하차'].map(lambda x : x**2).hist()
min(df1['일평균_승하차'])
df1.loc[df1['일평균_승하차'] == 0, :]

from sklearn.preprocessing import MinMaxScaler as minmax
m_mm = minmax()
m_mm()


# 3) 전체적 분포 확인
station_df_3_x.hist()    # 전체적 분포가 왼쪽에 치우침
pd.plotting.scatter_matrix(station_df_3_x, s = 25)

# 로그 변환
df_log_avg = np.log(station_df_3_x['일평균승하차'])
plt.boxplot(df_log_avg)

import seaborn as sns
sns.kdeplot(df_log_avg)     # log 변환한 데이터 KDE plot 확인 => 정규분포에 가깝게 출력됨

station_df_3_x['주민등록인구수'].hist()
df_log_pop = np.log(station_df_3_x['주민등록인구수'])
sns.kdeplot(df_log_pop)    # 비교적 정규분포에 가깝게 나옴

df4 = station_df_3_x.iloc[:, [0,1,4,5]]
df4['주민등록인구수'] = df_log_pop
df4['일평균승하차'] = df_log_avg
df_log_avg.describe()
df4.describe()
df4['주민등록인구수'].describe()
df4['일평균승하차'].describe()
df4.loc[df4['일평균승하차'] == df4['일평균승하차'].min(), '일평균승하차'] = df4.drop(77)['일평균승하차'].min()
df5 = df4.drop('스크린수', axis=1)

from sklearn.preprocessing import StandardScaler as standard
# 1) model 생성
m_st = standard()

# 2) 훈련 (학습 / fitting)
station_st_x = m_st.fit_transform(df5)

from sklearn.model_selection import train_test_split
st_tr_x, st_te_x, st_tr_y, st_te_y = train_test_split(df5, station_df_3_y, random_state=0)

from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(st_tr_x, st_tr_y)
m_lr.score(st_tr_x, st_tr_y)          # 56.95
m_lr.score(st_te_x, st_te_y)


from sklearn.ensemble import RandomForestRegressor as rf_r
m_rfr = rf_r()
m_rfr.fit(st_tr_x, st_tr_y)
m_rfr.score(st_tr_x, st_tr_y)     # 85.06
m_rfr.score(st_te_x, st_te_y)     # 28.22


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, 
                init = 'k-means++')   # default : 관측치들 중 초기값 선언
kmeans.fit(df5)

kmeans_cluster = kmeans.labels_
Series(kmeans_cluster).value_counts()


df5
df_grade1 = df_grade.drop([78,82,102,117,118,124,138,183])
df_grade1.reset_index(drop=True, inplace=True)
df_grade.loc[:, ['']]









