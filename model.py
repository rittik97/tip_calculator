import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import pickle

data=pd.read_csv('data/tips.csv')
data.head()

data['tip_p']=data['tip']/data['total_bill']
data['bill_per_person']=data['total_bill']/data['size']
# Removing Outliers
data=data[data['tip_p']<=0.35]




data.info()


data.describe().T

def naive_pred(X):
    #X['pred'] = X['total_bill']*0.1574
    return X['total_bill']*0.1574

temp=data[['tip','total_bill']]
y=temp['tip']
X=temp.drop('tip', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
r2_score(y_test,naive_pred(X_test))
mean_squared_error(y_test,naive_pred(X_test))**0.5
'''
plt.hist(data['tip'], bins=10);
len(data[data['tip']>6])
plt.hist(data['tip_p'], bins=10);

plt.hist(data['bill_per_person'], bins=10);

len(data[data['bill_per_person']>15])

def f(x):
    d = {}
    d['mean_bill'] = x['total_bill'].mean()
    d['tip'] = x['tip'].mean()
    d['tip_std'] = x['tip'].std()
    d['tip_p'] = x['tip_p'].mean()
    d['size'] = x['size'].mean()
    d['count']=len(x)
    d['count%']=len(x)/len(data)
    d['per_person_avg']=x['bill_per_person'].mean()
    return pd.Series(d)


data.groupby('sex').apply(f)

data.groupby('smoker').apply(f)

data.groupby('day').apply(f)

data.groupby('time').apply(f)

data.groupby('size').apply(f)

len(data[(data['size']==1) | (data['size']==5) | (data['size']==6)])

data.groupby(['time','day']).apply(f)

#sns.heatmap(data.corr(), annot=True)

#sns.pairplot(data[['tip','total_bill','size']])
'''
mapper = DataFrameMapper([
    ('sex', LabelEncoder()),
    ('smoker', LabelEncoder()),
    ('day', LabelBinarizer()),
    (['total_bill'], None),
    ('time', LabelEncoder()),
    ('tip', None)

], df_out=True)

temp=mapper.fit_transform(data)

#sns.heatmap(temp.corr(), annot=True)




temp=data[
    (data['bill_per_person']<15) |
    (data['tip']<6) |
    (data['size']!=1) |
    (data['size']!=5) |
    (data['size']!=6) |
    (data['tip_p']<0.35)

]



#data.head()

#temp=data

y=temp['tip']
X=temp.drop('tip', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

mapper = DataFrameMapper([
    ('sex', LabelEncoder()),
    ('smoker', LabelEncoder()),
    ('day', LabelBinarizer()),
    (['total_bill'], StandardScaler()),
    ('time', LabelEncoder()),
    ('size',None)
], df_out=True)

X_train.head()
#mapper.fit(X_train)

pf=PolynomialFeatures(interaction_only=True)
#pf.fit(X_train)

pca = PCA(n_components=4)
#pca.fit(X_train)
#pca.explained_variance_ratio_.sum()
#pca.explained_variance_ratio_

#Z_train=pca.transform(X_train)


feature_pipe=make_pipeline(mapper,pf, pca)
Z_train=feature_pipe.fit_transform(X_train)
Z_test=feature_pipe.fit_transform(X_test)

model=LassoCV(n_jobs=-1, n_alphas=10000 ).fit(Z_train,y_train)
model.alpha_
y_pred=model.predict(Z_test)
model.score(Z_train,y_train)
r2_score(y_test,y_pred)
mean_squared_error(y_test,y_pred)**0.5


'''



grid = GridSearchCV(estimator=ElasticNet(max_iter=10000),
                    param_grid={'alpha': [0.007,0.003,0.005,0.009],
                                'l1_ratio': [25, 30, 35],
                                'fit_intercept':[True,False]
                                },
                    cv=5,
                    verbose = 1,
                    return_train_score = True)

grid.fit(Z_train,y_train)
grid.best_params_
grid.best_score_
r2_score(y_test,grid.best_estimator_.predict(Z_test))
mean_squared_error(y_test,grid.best_estimator_.predict(Z_test))**0.5

grid = GridSearchCV(estimator=DecisionTreeRegressor(),
                    param_grid={'min_samples_split': [2,3,4],
                                'max_depth': [None,3,4,5],
                                },
                    cv=5,
                    verbose = 1,
                    return_train_score = True)

grid.fit(Z_train,y_train)
grid.best_params_
grid.best_score_
r2_score(y_test,grid.best_estimator_.predict(Z_test))
mean_squared_error(y_test,grid.best_estimator_.predict(Z_test))**0.5


from catboost import CatBoostRegressor
cat_features=['sex','smoker','day','time']
y=data['tip']
X=data.drop(['tip','tip_p'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = CatBoostRegressor(cat_features=cat_features).fit(X_train,y_train)
model.score(X_train,y_train)
model.score(X_test,y_test)
mean_squared_error(y_test,model.predict(X_test))**0.5
'''
model
pipe=make_pipeline(feature_pipe,model)
pickle.dump(pipe, open("pipe.pkl", "wb"))


