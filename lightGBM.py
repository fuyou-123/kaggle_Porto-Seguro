import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc
import os
from multiprocessing import *

#### Load Data
os.getcwd()
os.chdir('/Users/miao/Desktop/kaggle')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

###
y = train['target'].values
testid = test['id'].values

train.drop(['id', 'target'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

### Drop calc
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(unwanted, axis=1)
test = test.drop(unwanted, axis=1)



def recon(reg):
    integer = int(np.round((40 * reg) ** 2))
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A) // 31
    return A, M


train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
train['ps_reg_A'].replace(19, -1, inplace=True)
train['ps_reg_M'].replace(51, -1, inplace=True)
test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
test['ps_reg_A'].replace(19, -1, inplace=True)
test['ps_reg_M'].replace(51, -1, inplace=True)



d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
d_skew = train.skew(axis=0)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id', 'target']}


def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id', 'target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol] == -1).values, axis=1)
    for c in dcol:
        if '_bin' not in c:  # standard arithmetic
            df[c + str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c + str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c]) > 2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c + '_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df


def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close();
    p.join()
    print('After Shape: ', df.shape)
    return df


train = multi_transform(train)
test = multi_transform(test)

features=train.columns




# custom objective function (similar to auc)

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)


def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True




X = train.values

sub = pd.DataFrame()
sub['id'] = testid
sub['target']=np.zeros((len(testid),))





# lgb
params = {'metric': 'auc', 'learning_rate': 0.01, 'max_depth': 5, 'max_bin': 10, 'objective': 'binary',
          'feature_fraction': 0.8, 'bagging_fraction': 0.9, 'bagging_freq': 10, 'min_data': 500}

kfold=5
nrounds=3000
skf = StratifiedKFold(n_splits=kfold, random_state=1)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' lgb kfold: {}  of  {} : '.format(i + 1, kfold))
    X_train, X_eval = X[train_index], X[test_index]
    y_train, y_eval = y[train_index], y[test_index]
    lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds,
                          lgb.Dataset(X_eval, label=y_eval), verbose_eval=100,
                          feval=gini_lgb, early_stopping_rounds=1000)
    sub['target'] += lgb_model.predict(test[features].values,
                                       num_iteration=lgb_model.best_iteration) / ( kfold)

sub.to_csv('lgb.csv', index=False, float_format='%.5f')
gc.collect()
