import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import *
import gc
import os
import matplotlib.pyplot as plt
import operator
import pywFM
from numba import jit


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

### Gini

@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini



# part 1

increase = True

if increase:
    # Get positive examples
    pos = pd.Series(y == 1)
    # Add positive examples
    pos.index=train.index
    y = pd.DataFrame(y)
    y.index = train.index
    train = pd.concat([train, train.loc[pos]], axis=0)
    # y1.shape=(len(y1),1)
    y = pd.concat([y, y.loc[pos]], axis=0)
    # Shuffle data
    idx = np.arange(len(train))
    np.random.shuffle(idx)
    train = train.iloc[idx]
    y = y.iloc[idx]


from sklearn.preprocessing import MinMaxScaler
def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X


# PCA

Train = scale_data(train)
Test = scale_data(test)

from sklearn.decomposition import PCA
pca=PCA(20,svd_solver='full')
train_pca = pca.fit_transform(Train)
test_pca = pca.fit_transform(Test)
print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())

train_pca=pd.DataFrame(train_pca)
train_pca.index=train.index
test_pca=pd.DataFrame(test_pca)
test_pca.index=test.index

train=pd.concat([train,train_pca],axis=1)
test=pd.concat([test,test_pca],axis=1)

# pywfm
clf=pywFM.FM(task='classification',
                            num_iter = 1000,
                            init_stdev = 0.1,
                            k2 = 5,
                            learning_method = 'mcmc',
                            verbose = False,
                            silent = False)

y=np.asarray(y)
y.shape=(len(y), )

sub = pd.DataFrame()
sub['id'] = testid

y1=np.zeros((len(testid),))

model = clf.run(x_train=train, y_train=y, x_test=test, y_test=y1)

sub['target'] = model.predictions
sub.to_csv('libFM.csv', index=False)



# part 2

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validationMCMC_demo(data,target):
    # 1. Setting the lists of values on which we're doing the grid search as well as general parameters
    num_iter = 1000;
    k_fold = 5
    std_init_vec = [0.01,0.05,0.1,0.2];

    # 2. Loading the data and preparing the run.
    k_indices = build_k_indices(data, k_fold, 12)


    gini_te = []

    print("\t\tStarting the ", k_fold, "-fold cross validation for MCMC\n")
    for std_init in std_init_vec:
        gini = []
        for k_cv in range(0, k_fold):
            gini.append(cross_validationMCMC(data,target, k_indices, k_cv, num_iter, std_init))

        mean_gini = np.mean(gini)
        print("gini = ", mean_gini, " (for MCMC with with ", num_iter, "iterations, std_init =", std_init, ")")

        gini_te.append(mean_gini)

    best_gini=max(gini_te)

    best_std=std_init_vec[gini_te.index(max(gini_te))]

    return best_gini, best_std


def cross_validationMCMC(data,target, k_indices, k, num_iter, std_init):
    """
        Runs the cross validation on the input data, using the Markov Chain Monte Carlo algorithm.
        It splits the data into a training and testing fold, according to k_indices and k, and then runs
        the MCMC on all the parameter std_init for num_iter iterations.
        @param data : the DataFrame containing all our training data (on which we do the CV)
        @param k_indices : array of k-lists containing each of the splits of the data
        @param k : the number of folds of the cross-validation
        @param num_iter : the number of iterations of the algorithm
        @param std_init : the standard deviation for the initialisation of the data
        @return loss_te : the RMSE loss for the run of the algorithm using libFM with these parameters.

    """
    # get k'th subgroup in test, others in train
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indices = tr_indices.reshape(-1)

    x1 = data.loc[tr_indices]
    x2 = data.loc[te_indices]
    y1 = target[tr_indices]
    y2 = target[te_indices]


    # running the model
    fm = pywFM.FM(task='classification', num_iter=num_iter, init_stdev=std_init)

    model = fm.run(x1, y1, x2, y2)

    # getting the RMSE at the last run step.
    pred=model.predictions

    return eval_gini(pred,y2)


r,t=cross_validationMCMC_demo(train,y)

sub = pd.DataFrame()
sub['id'] = testid

fm = pywFM.FM(task='classification', num_iter=1000, init_stdev=t)


y1=np.zeros((len(testid),))


model = fm.run(train, y, test, y1)

sub['target'] = model.predictions
sub.to_csv('libFM.csv', index=False)
