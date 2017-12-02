import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
from collections import Counter
import plotly.graph_objs as go
import plotly.tools as tls
import os
import operator
from operator import itemgetter, attrgetter



os.getcwd()
os.chdir('/Users/miao/Desktop/kaggle')
train = pd.read_csv('train.csv', na_values=-1)
#print(train.head())



train = train.replace(-1, np.NaN)
train.columns
print(Counter(train.dtypes.values))
train_float = train.select_dtypes(include=['float64'])
train_int = train.select_dtypes(include=['int64'])




# target
train['target'].describe()
# plot
data = [go.Bar(
               x = train["target"].value_counts().index.values,
               y = train["target"].value_counts().values,
               text='Distribution of target variable'
               )]

layout = go.Layout(
                   title='Target variable distribution'
                   )

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='basic-bar')


# variable (the value of variable is integer)
string="ps_ind_03"
train[string].describe()
u=train[string].unique()
sorted(u)

# plot
v=train[string]*(1-train["target"])
Yy=v.value_counts().values
kk=np.zeros((len(Yy),2),dtype=np.int)  # for variable whose unique value are rational number, delete "dtype=np.int"
kk[:,0]=np.asarray(pd.DataFrame(v.value_counts()).index.tolist())
kk[:,1]=Yy
for i in range(len(u)):
    if u[i] not in kk[:,0]:
        if not math.isnan(u[i]):
            kk=np.vstack((kk,[u[i],0]))

Yy=sorted(kk, key=itemgetter(0))
Yy=np.array(Yy)
Yy=Yy[:,1]
if 0 in kk[:,0]:
    indicator=(train[string]==0)*(1-train["target"])
    Yy[0]=sum(indicator)


v=train[string]*train["target"]
Y=v.value_counts().values
kk=np.zeros((len(Y),2),dtype=np.int) # for variable whose unique value are rational number, delete "dtype=np.int"
kk[:,0]=np.asarray(pd.DataFrame(v.value_counts()).index.tolist())
kk[:,1]=Y
for i in range(len(u)):
    if u[i] not in kk[:,0]:
        if not math.isnan(u[i]):
            kk=np.vstack((kk,[u[i],0]))

Y=sorted(kk, key=itemgetter(0))
Y=np.array(Y)
Y=Y[:,1]
if 0 in kk[:,0]:
    indicator=(train[string]==0)*train["target"]
    Y[0]=sum(indicator)

if Yy[0]==Y[0]==0:
    Yy=Yy[1:]
    Y=Y[1:]


data1 = go.Bar(
               x = sorted(u),
               y = Y,
               name='target 1'
               )

data2 = go.Bar(
               x = sorted(u),
               y = Yy,
               name='target 0'
               )
data=[data1,data2]

layout = go.Layout(
                   barmode='stack',
                   title=string
                   )

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='basic-bar')

Yy/Y


# variable (the value of variable is rational variable)
string="ps_car_15"
train[string].describe()
u=train[string].unique()
sorted(u)
# plot
v=train[string]*(1-train["target"])
Yy=v.value_counts().values
kk=np.zeros((len(Yy),2))
kk[:,0]=np.asarray(pd.DataFrame(v.value_counts()).index.tolist())
kk[:,1]=Yy
for i in range(len(u)):
    if u[i] not in kk[:,0]:
        if not math.isnan(u[i]):
            kk=np.vstack((kk,[u[i],0]))

Yy=sorted(kk, key=itemgetter(0))
Yy=np.array(Yy)
Yy=Yy[:,1]
if 0 in kk[:,0]:
    indicator=(train[string]==0)*(1-train["target"])
    Yy[0]=sum(indicator)


v=train[string]*train["target"]
Y=v.value_counts().values
kk=np.zeros((len(Y),2))
kk[:,0]=np.asarray(pd.DataFrame(v.value_counts()).index.tolist())
kk[:,1]=Y
for i in range(len(u)):
    if u[i] not in kk[:,0]:
        if not math.isnan(u[i]):
            kk=np.vstack((kk,[u[i],0]))

Y=sorted(kk, key=itemgetter(0))
Y=np.array(Y)
Y=Y[:,1]
if 0 in kk[:,0]:
    indicator=(train[string]==0)*train["target"]
    Y[0]=sum(indicator)

if Yy[0]==Y[0]==0:
    Yy=Yy[1:]
    Y=Y[1:]


data1 = go.Bar(
               x = sorted(u),
               y = Y,
               name='target 1'
               )

data2 = go.Bar(
               x = sorted(u),
               y = Yy,
               name='target 0'
               )
data=[data1,data2]

layout = go.Layout(
                   barmode='stack',
                   title=string
                   )

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='basic-bar')


Yy/Y


# boxplot
ax=sns.boxplot(x=train['target'],y=train[string])
ax = sns.stripplot(x=train['target'],y=train[string],jitter=True,color=".3")











# any() applied twice to check run the isnull check across all columns.
print(train.isnull().any().any())

# missing data
aa=train_copy.isnull().sum()/train_copy.shape[0]
train = pd.read_csv("train.csv",header=None)
train=np.array(train)
variable=train[0,:]
variable=np.transpose(variable)
#print(variable)
aa=np.vstack((variable,aa))
aa=pd.DataFrame(aa)
aa=np.transpose(aa)


aa.to_csv('variable.csv', index=False)

