# -*- coding: utf-8 -*-

df2

com_1 = pd.read_csv('사업체수2.csv')
com_2 = com_1.groupby('코드')[['사업체수']].sum().astype('int')
com_2.reset_index(inplace=True)

com_2.columns = ['ADM_CD', '사업체수']
com_2.to_csv('행정구별_사업체수.csv', encoding='UTF-8')

df_code = pd.read_table('행정구역코드통합_1.txt', sep=',')
df_code = df_code.drop('인덱스', axis=1)
df_code = df_code.drop(77)
df_code.reset_index(drop=True, inplace=True)
df_code = df_code.drop([77, 81,101,116,117,123,137,182])
df2['행정구역코드'] = df_code['지역코드']


df2
com_2


# =============================================================================
# qgis shp파일 수정
# =============================================================================
import geopandas as gpd

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['figure.figsize'] =(10,10)

seoul_file = 'D:/03_Python/04_프로젝트/qgis/230411/new_shape_01.shp'
seoul = gpd.read_file(seoul_file, encoding = 'UTF-8')

df_pop = pd.read_table('D:/03_Python/04_프로젝트/qgis/격자/2020년_인구_다사_1K_work.txt', sep='^',
                       header = None)

df_pop = df_pop.drop([0, 2], axis=1)
df_pop.columns = ['행정구역', '인구수']
df_pop_1 = df_pop.groupby('행정구역')[['인구수']].sum()
df_pop_1 = df_pop_1.reset_index()
df_pop_1.to_csv('2020년_인구_다사_1K_work.csv', encoding='UTF-8')












