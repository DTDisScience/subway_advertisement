# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:03:09 2023

@author: itwill
"""

# =============================================================================
# 군집분석
# =============================================================================

# 1) 데이터불러오기

dff = pd.read_csv('군집분석데이터.csv')

dff = dff.set_index('역명')
line = dff['호선']
dff = dff.drop('호선',axis=1)

# 2) 스케일링

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

mm = MinMaxScaler()
df_mm = mm.fit_transform(dff)
df_mm = pd.DataFrame(df_mm, columns=dff.columns)

sc = StandardScaler()
df_sc = sc.fit_transform(dff)

# 3) 군집분석
from sklearn.cluster import KMeans
from sklearn.cluster import dbscan

# 3-1) KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(dff)
kmeans_cluster = kmeans.labels_
Series(kmeans.labels_).value_counts()

# 3-2) dbscan
_, dbscan_cluster = dbscan(df_mm, eps = 0.4, min_samples = 2)
Series(dbscan_cluster).value_counts()
#  0    187
# -1     11
#  1      3
# -1,과 1을 같은 등급(높은 등급)으로 취급.



df3.head()
df3['등급'].value_counts()

df3['예측'] = kmeans.labels_

df3.loc[df3['등급']==2,'등급'] = 0

df3[df3['등급']==df3['예측']]
df3.loc[df3['등급']!=df3['예측'],'역명']

df3.loc[df3['예측']==1,'역명']
df3.loc[df3['예측']==1,:]

df3.loc[df3['역명']=='건대입구']
df3.loc[df3['역명']=='고속터미널']

df3['dbscan'] = dbscan_cluster
df3.loc[df3['dbscan']==-1,'dbscan'] = 1

df3.loc[df3['등급']==df3['dbscan'],'역명']
df3.loc[df3['dbscan']==1,'역명']
df3.loc[df3['등급']!=df3['dbscan'],'역명']

df3['역명'].value_counts() > 1
df3['역명'].value_counts()>2

df3[df3['역명'].value_counts() > 1]
list(df3['역명'].value_counts() > 1)
count_series = df3['역명'].value_counts() > 1
name = count_series.name

transfer = df3['역명'].value_counts()[df3['역명'].value_counts() > 1].index.tolist()

df4 = df3[df3['역명'].isin(transfer)]

df4.sort_values('역명')[['역명','호선','등급','예측','dbscan']]

df3.loc[df3['등급'] == 1,'역명']
df3.loc[df3['등급'] == 0,'역명']

df3[df3['역명'].isin(transfer)]

# k-means 사용 시 환승역으로 인한 오분류 예상역
# '건대입구','고속터미널','사당','잠실','

# 시각화
dff['kmeans'] = df3['예측']
dff['dbscan'] = df3['dbscan']


plt.plot(df_mm['스타벅스수'][df_mm['kmeans'] == 1], df_mm['스타벅스수'][df_mm['kmeans'] == 1], 'o', label='kmeans 1')
plt.plot(df_mm['대합실면적'][df_mm['kmeans'] == 0], df_mm['대합실면적'][df_mm['kmeans'] == 0], 'o', label='kmeans 2')

plt.plot(dff['스타벅스수'][dff['kmeans'] == 1], dff['스타벅스수'][dff['kmeans'] == 1], 'o', label='kmeans 1')
plt.plot(dff['대합실면적'][dff['kmeans'] == 0], dff['대합실면적'][dff['kmeans'] == 0], 'o', label='kmeans 2')

plt.plot(df_mm['퇴근시간대일반인승차'][df_mm['kmeans'] == 1], df_mm['퇴근시간대일반인승차'][df_mm['kmeans'] == 1], 'o', label='kmeans 1',alpha=0.3)
plt.plot(df_mm['출근시간대하차인원'][df_mm['kmeans'] == 0], df_mm['출근시간대하차인원'][df_mm['kmeans'] == 0], 'o', label='kmeans 2',alpha=0.3)

cor = dff.corr()
sns.heatmap(cor,annot=True)

sns.heatmap(df4.iloc[:,2:].corr(),annot=True)