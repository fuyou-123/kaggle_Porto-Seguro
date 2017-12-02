import pandas as pd
import numpy as np
import re
import sklearn
import plotly.graph_objs as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import os

os.getcwd()
os.chdir('/Users/miao/Desktop/kaggle')



from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold



print('loading files...')
train = pd.read_csv('train.csv', na_values=-1)



#sub['parameter'] = 0
test = pd.read_csv('test.csv', na_values=-1)
#col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
#train = train.drop(col_to_drop, axis=1)
#test = test.drop(col_to_drop, axis=1)

for c in train.select_dtypes(include=['float64']).columns:
    train[c]=train[c].astype(np.float32)
    test[c]=test[c].astype(np.float32)
for c in train.select_dtypes(include=['int64']).columns[2:]:
    train[c]=train[c].astype(np.int8)
    test[c]=test[c].astype(np.int8)

print(train.shape, test.shape)

y = train['target'].values
sub = test['id'].to_frame()
sub['target'] = 0
X = train.drop(['id', 'target'], axis=1)
test = test.drop(['id'], axis=1)


cat_cols = [col for col in X.columns if 'cat' in col]
bin_cols = [col for col in X.columns if 'bin' in col]
con_cols = [col for col in X.columns if col not in bin_cols + cat_cols]

for col in cat_cols:
    X[col].fillna(value=X[col].mode()[0], inplace=True)
    X[col].fillna(value=X[col].mode()[0], inplace=True)

for col in bin_cols:
    X[col].fillna(value=X[col].mode()[0], inplace=True)
    test[col].fillna(value=test[col].mode()[0], inplace=True)

for col in con_cols:
    X[col].fillna(value=X[col].mean(), inplace=True)
    test[col].fillna(value=test[col].mean(), inplace=True)


cat_features = [a for a in X.columns if a.endswith('cat')]


for column in cat_features:
    temp = pd.get_dummies(pd.Series(X[column]), prefix = column)
    X = pd.concat([X,temp], axis=1)
    X = X.drop([column], axis=1)



cat_features = [a for a in test.columns if a.endswith('cat')]


for column in cat_features:
    temp = pd.get_dummies(pd.Series(test[column]), prefix = column)
    test = pd.concat([test,temp], axis=1)
    test = test.drop([column], axis=1)



# custom objective function (similar to auc)

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True


XX=X
features = X.columns
X = X.values
print(X.shape)







nrounds = 2000  # need to change to 2000
kfold = 5  # need to change to 5


rf = RandomForestClassifier(nrounds)
rf.fit(X,y)
sub['target'] += rf.predict_proba(test[features].values)[:,1]


sub.to_csv('rf_1.csv', index=False, float_format='%.5f')

rf.feature_importances_


sub1=np.zeros((features.shape[0]))
sub=np.string_((features.shape[0]))
sub=features
sub=np.vstack((sub,sub1))
sub=sub.transpose()
sub[:,1]=rf.feature_importances_
sub=pd.DataFrame(sub)
sub.to_csv('rf_importance.csv', index=False, float_format='%.5f')
#sub.to_csv('lasso.csv', index=False, float_format='%.5f')
#print(searchCV.coef_)
#print(features)


# Scatter plot
trace = go.Scatter(
    y = searchCV.coef_,
    x = features,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 13,
        #size= rf.feature_importances_,
        #color = np.random.randn(500), #set color equal to a variable
        color = searchCV.coef_,
        colorscale='Portland',
        showscale=True
    ),
    text = features
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig,filename='scatter2010')
