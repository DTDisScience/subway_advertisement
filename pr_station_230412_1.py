# -*- coding: utf-8 -*-



group_dis = pd.read_csv('grid_distance.csv', encoding='UTF-8')
group_dis= group_dis.drop_duplicates(ignore_index = True)
dis_df1 = group_dis.loc[:, ['전철역명', '호선', '가까운GRID_1K_CD', '가까운2020년_인구_다사_1K_work_인구수',
                  '가까운cnt_company_사업체수', '가까운cnt_worker_종사자수', 'n', 'distance']]
dis_df1.loc[(dis_df1['전철역명'] == '가락시장') & (dis_df1['호선'] == '3') , :]

dis_df1 = dis_df1.drop('가까운GRID_1K_CD', axis=1)
dis_df1.columns = ['역명', '호선', '인구수', '사업체수', '종사자수', '군집', '거리']
dis_df2 = dis_df1.groupby(['역명', '호선'])[['인구수','사업체수', '종사자수']].sum()
dis_df2.reset_index(inplace=True)
dis_df2

con1 = ['1', '2', '3', '4', '5', '6', '7', '8']
dis_df2['호선'] = dis_df2['호선'].map(lambda x : int(x) if x in con1 else 9)
dis_df3 = dis_df2.loc[dis_df2['호선'] != 9, :]

dis_df3.reset_index(drop=True, inplace=True)
dis_df3



dis_df3['역명'] = list(map(name_change, dis_df3['역명'], dis_df3['호선']))
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

dis_df4 = dis_df3.sort_values(['호선', '역명'], ascending = [True,True])
dis_df4.reset_index(drop=True, inplace = True)
dis_df4


con2 = [1, 2, 3, 4, 5, 7, 8]
vres = []
for i in con2:
    condi1 = condi_df.loc[condi_df['호선'] == i, '역명'].values
    v1 = dis_df4.loc[dis_df4['호선'] == i, '역명'].index
    for j in v1 :
        if dis_df4.iloc[j, 0] in condi1 :
            vres.append(j)

dis_df5 = dis_df4.iloc[vres, :]
dis_df5.reset_index(drop=True, inplace = True)
dis_df5.iloc[:, 2:].hist()
grid4 = np.log1p(dis_df5.iloc[:, 3:])
grid4

df6['사업체수'] = grid4['사업체수']
df6['종사자수'] = grid4['종사자수']

df6
# =============================================================================
# train/test 분리  (df6; df_y 로 진행)
# =============================================================================
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df6.values, df_y, random_state=0)

# =============================================================================
# 로지스틱 회귀 모델링
# =============================================================================
from sklearn.linear_model import LogisticRegression as lr
m_lr = lr()
m_lr.fit(train_x, train_y)
m_lr.score(train_x, train_y)           # 0.7266666666666667
m_lr.score(test_x, test_y)             # 0.7647058823529411

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
np.mean(train_score1)        # 0.7445584988962473
np.mean(test_score1)         # 0.6861764705882354


from sklearn.tree import DecisionTreeClassifier as dt_c
m_dt = dt_c()
m_dt.fit(train_x, train_y)
m_dt.score(train_x, train_y)
m_dt.score(test_x, test_y)
m_dt.feature_importances_
array([0.10769322, 0.40029696, 0.089928  , 0.24895228, 0.10471875,
       0.04841079])
df6.columns
# '일평균승하차', '대합실면적', '밀도', '사업체수', '종사자수', '주민등록인구수'
# 0.10769322, 0.40029696, 0.089928  , 0.24895228, 0.10471875, 0.04841079



grid5 = dis_df5.iloc[:, 2:]
df2['주민등록인구수'] = grid5['인구수'].map(lambda x : int(x))
df2['사업체수'] = grid5['사업체수'].map(lambda x : int(x))
df2['종사자수'] = grid5['종사자수'].map(lambda x : int(x))
df1.iloc[:, :2]
df2
pd.concat([df1.iloc[:, :2], df2], axis=1).to_csv('grid_total.csv', encoding='UTF-8')

# =============================================================================
# 선형회귀
# =============================================================================
df1 = pd.read_csv('grid_total.csv', encoding='UTF-8')
df2 = df1.iloc[:, 3:9]
df_y = df1['등급']
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df2.values, df_y, random_state=0)

from sklearn.linear_model import LinearRegression as reg
m_reg = reg()
m_reg.fit(train_x, train_y)
m_reg.score(train_x, train_y)
m_reg.score(test_x, test_y)

m_reg.coef_
m_reg.intercept_



































