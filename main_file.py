#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:36:35 2019

@author: jessica.troianello
"""

# Load libraries and check memory

import psutil;

from GAN1 import SimpleAccuracy, SimpleMetrics, adversarial_training_GAN

print(list(psutil.virtual_memory())[0:2])

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use('ggplot')

import xgboost as xgb

import pickle

import gc

gc.collect()
print(list(psutil.virtual_memory())[0:2])

import GAN1

# For reloading after making changes
import importlib

# importlib.reload(GAN_171103)
# from GAN_171103 import *

# Load engineered dataset from EDA section

# data = pickle.load(open('data/' + 'credicard.engineered.pkl','rb'))
#
## data columns will be all other columns except class
# data_cols = list(data.columns[ data.columns != 'Class' ])
# label_cols = ['Class']
#
# print(data_cols)
# print('# of data columns: ',len(data_cols))
#      
## Put columns in order of importance for xgboost fraud detection (from that section)
#
## sorted_cols = ['V14', 'V4', 'V12', 'V10', 'V26', 'V17', 'Amount', 'V7', 'V21', 'V28', 'V20', 'V3', 'V18', 'V8', 'V13', 'V22', 'V16', 'V11', 'V19', 'V27', 'V5', 'V6', 'V25', 'V15', 'V24', 'V9', 'V1', 'V2', 'V23', 'Class']
## sorted_cols = ['V14', 'V4', 'V12', 'V10', 'Amount', 'V26', 'V17', 'Time', 'V7', 'V28', 'V21', 'V19', 'V8', 'V3', 'V22', 'V20', 'V25', 'V11', 'V6', 'V16', 'V27', 'V5', 'V18', 'V9', 'V1', 'V2', 'V15', 'V23', 'V24', 'V13', 'Class']
# sorted_cols = ['V14', 'V4', 'V10', 'V17', 'Time', 'V12', 'V26', 'Amount', 'V21', 'V8', 'V11', 'V7', 'V28', 'V19', 'V3', 'V22', 'V6', 'V20', 'V27', 'V16', 'V13', 'V25', 'V24', 'V18', 'V2', 'V1', 'V5', 'V15', 'V9', 'V23', 'Class']
#
# data = data[ sorted_cols ].copy()
#
## Add KMeans generated classes to fraud data - see classification section for more details on this
#
# import sklearn.cluster as cluster
#
# train = data.loc[ data['Class']==1 ].copy()
#
# algorithm = cluster.KMeans
# args, kwds = (), {'n_clusters':2, 'random_state':0}
# labels = algorithm(*args, **kwds).fit_predict(train[ data_cols ])
#
# print( pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) ) )
#
# fraud_w_classes = train.copy()
# fraud_w_classes['Class'] = labels
#
#
# def create_toy_spiral_df( n, seed=0):
#    np.random.seed(seed)
#    toy = np.array([ [ (i/10+1) * np.sin(i), -(i/10+1) * np.cos(i) ] for i in np.random.uniform(0,3*np.pi,size=n) ])
#    toy = pd.DataFrame( toy, columns=[ ['v'+str(i+1) for i in range(2)] ])
#    return toy
#
## toy = create_toy_spiral_df(1000)    
## plt.scatter( toy['v1'], toy['v2'] ) ;
#
## Function to create toy dataset of multiple groups of normal distributions in n dimensions
#
# def create_toy_df( n, n_dim, n_classes, seed=0):
#    toy = pd.DataFrame(columns=[ ['v'+str(i+1) for i in range(n_dim)] + ['Class'] ])
#    toy_cols = toy.columns
#    np.random.seed(seed)
#    for class0 in range(n_classes):
#        center0s = np.random.randint(-10,10,size=n_dim)/10
#        var0s = np.random.randint(1,3,size=n_dim)/10
#        temp = np.array([[class0]]*n)
#        for dim0 in range(n_dim):
#            temp = np.hstack( [np.random.normal(center0s[dim0],var0s[dim0],n).reshape(-1,1), temp] )
#        toy = pd.concat([toy,pd.DataFrame(temp,columns=toy_cols)],axis=0).reset_index(drop=True)
#    return toy
#
## toy = create_toy_df(n=1000,n_dim=2,n_classes=2,seed=0)
## plt.scatter(toy[toy.columns[0]],toy[toy.columns[1]],c=toy['Class'], alpha=0.2) ;

# Load the credit card data

# Original data available from:
# https://www.kaggle.com/dalpozz/creditcardfraud

data = pd.read_csv("data/creditcard.csv")
print(data.shape)
print(data.columns)
data.head(3)

# data columns will be all other columns except class

label_cols = ['Class']
data_cols = list(data.columns[data.columns != 'Class'])

print(data_cols)
print('# of data columns: ', len(data_cols))

# 284315 normal transactions (class 0)
# 492 fraud transactions (class 1)

data.groupby('Class')['Class'].count()

# Total nulls in dataset (sum over rows, then over columns)

data.isnull().sum().sum()

# Duplicates? Yes

normal_duplicates = sum(data.loc[data.Class == 0].duplicated())
fraud_duplicates = sum(data.loc[data.Class == 1].duplicated())
total_duplicates = normal_duplicates + fraud_duplicates

print('Normal duplicates', normal_duplicates)
print('Fraud duplicates', fraud_duplicates)
print('Total duplicates', total_duplicates)
print('Fraction duplicated', total_duplicates / len(data))

# 'Time' is seconds from first transaction in set
# 48 hours worth of data
# Let's convert time to time of day, in hours

print('Last time value: {:.2f}'.format(data['Time'].max() / 3600))

data['Time'] = (data['Time'].values / 3600) % 24

plt.hist([data.loc[data['Class'] == 0, 'Time'], data.loc[data['Class'] == 1, 'Time']],
         normed=True, label=['normal', 'fraud'], bins=np.linspace(0, 24, 25))
plt.legend()
plt.title('Fraud .vs Non Fraud by Time of Day')
plt.xlabel('Time of Day')

plt.show()
plt.savefig('plots/Plot1.png')
# Looks like normal transactions have a bias towards 8am to midnight
# Fraud has spikes at 2-3am and noon

# several columns heavily skewed, 'Amount' the highest (besides Class)

data.skew()

# Minimum 'Amount' is 0
# 0's account for 0.6% of the data set

print(data['Amount'].min())
print(np.sum(data['Amount'] == 0))
# print( np.sum( data['Amount']<0.01 ) )
print(np.sum(data['Amount'] == 0) / len(data))

# Looks like all 'Amount' values are rounded to the hundredths (0.01) place
data['Amount'].mod(0.01).hist()
plt.savefig('plots/Plot2.png')

# Some values are much more frequent than others
# 0.00 comes in 12th in the list

print(data.Amount.value_counts().head(15))

# Log transform amount values to give more normal distribution

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.hist(data['Amount'], bins=40)
plt.title('Original Amount Distribution')

plt.subplot(1, 2, 2)
d0 = np.log10(data['Amount'].values + 1)
# d0 = np.log1p( data['Amount'].values ) / np.log(10)
plt.hist(d0, bins=40)
plt.title('Log10(x+1) Transformed Amount Distribution')
plt.savefig('plots/Plot3.png')
plt.show()

# Use log transformed data

data['Amount'] = d0

# Center and scale all data, only using the middle 99.8%, so outliers don't pull too much.
# First generate the percentile data for each feature

percentiles = pd.DataFrame(np.array([np.percentile(data[i], [0.1, 99.9]) for i in data_cols]).T,
                           columns=data_cols, index=['min', 'max'])

percentile_means = \
    [[np.mean(data.loc[(data[i] > percentiles[i]['min']) & (data[i] < percentiles[i]['max']), i])]
     for i in data_cols]

percentiles = percentiles.append(pd.DataFrame(np.array(percentile_means).T, columns=data_cols, index=['mean']))

percentile_stds = \
    [[np.std(data.loc[(data[i] > percentiles[i]['min']) & (data[i] < percentiles[i]['max']), i])]
     for i in data_cols]

percentiles = percentiles.append(pd.DataFrame(np.array(percentile_stds).T, columns=data_cols, index=['stdev']))

percentiles

# Center and scale the data using the percentile data we just generated

data[data_cols] = (data[data_cols] - percentiles.loc['mean', data_cols]) / percentiles.loc['stdev', data_cols]

# # Or we can center and scale using all of the data

# from sklearn.preprocessing import StandardScaler

# data[data_cols] = StandardScaler().fit_transform(data[data_cols])

# There are outliers, 50-100 stdevs away from mean in several columns

plot_cols = data_cols
# plt.plot( np.log10( data[ plot_cols ].abs().max().values ) )
# plt.plot( data[ plot_cols ].abs().max().values / data[ plot_cols ].std().values / 10, label='max z/10' )
plt.plot(data.loc[data.Class == 1, plot_cols].abs().max().values / data[plot_cols].std().values / 10,
         label='fraud max z/10')
plt.plot(data.loc[data.Class == 0, plot_cols].abs().max().values / data[plot_cols].std().values / 10,
         label='real max z/10')
plt.plot(data[plot_cols].mean().values, label='mean')
# plt.plot( data[ plot_cols ].abs().mean().values, label='abs mean' )
plt.plot(data[plot_cols].std().values, label='std')
plt.xlabel('Feature')
plt.ylabel('z/10')
plt.legend()

plt.savefig('plots/Plot4.png')
plt.show()

# Check Correlations
# Note no correlations among PCA transformed columns, as expected
corr0 = data.corr()


plt.savefig('plots/Plot5.png')
plt.imshow(corr0)
# Looking at correlation values

# np.round(corr0[['Time','Amount','Class']],2)
# plt.imshow( np.round(corr0[['Time','Amount','Class']],2) ) ;
# corr0[data_cols]
# np.round(corr0[data_cols],1)
# np.round(corr0[data_cols],1)

# Plot the data by each feature

axarr = [[]] * len(data_cols)
columns = 4
rows = int(np.ceil(len(data_cols) / columns))
f, fig = plt.subplots(figsize=(columns * 3.5, rows * 2))

f.suptitle('Data Distributions by Feature and Class', size=16)

for i, col in enumerate(data_cols[:]):
    axarr[i] = plt.subplot2grid((int(rows), int(columns)), (int(i // columns), int(i % columns)))
    axarr[i].hist([data.loc[data.Class == 0, col], data.loc[data.Class == 1, col]], label=['normal', 'fraud'],
                  bins=np.linspace(np.percentile(data[col], 0.1), np.percentile(data[col], 99.9), 30),
                  normed=True)
    axarr[i].set_xlabel(col, size=12)
    axarr[i].set_ylim([0, 0.8])
    axarr[i].tick_params(axis='both', labelsize=10)
    if i == 0:
        legend = axarr[i].legend()
        legend.get_frame().set_facecolor('white')
    if i % 4 != 0:
        axarr[i].tick_params(axis='y', left='off', labelleft='off')
    else:
        axarr[i].set_ylabel('Fraction', size=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # xmin, ymin, xmax, ymax
# plt.savefig('plots/Engineered_Data_Distributions.png')

plt.savefig('plots/Plot6.png')
plt.show()
# Save engineered dataset for use in analysis
# Save as pickle for faster reload

pickle.dump(data, open('data/' + 'credicard.engineered.pkl', 'wb'))

# data.to_csv('data/' + 'credicard.engineered.csv.zip')

# define the columns we want to test on, in case we want to use less than the full set

test_cols = data.columns

# test_cols = data.columns[ data.columns != 'Amount' ]

print(len(test_cols))
print(test_cols)

# Define some custom metric functions for use with the xgboost algorithm
# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py

from sklearn.metrics import recall_score, precision_score, roc_auc_score


def recall(preds, dtrain):
    labels = dtrain.get_label()
    return 'recall', recall_score(labels, np.round(preds))


def precision(preds, dtrain):
    labels = dtrain.get_label()
    return 'precision', precision_score(labels, np.round(preds))


def roc_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'roc_auc', roc_auc_score(labels, preds)


# Set up the test and train sets

np.random.seed(0)

n_real = np.sum(data.Class == 0)  # 200000
n_test = np.sum(data.Class == 1)  # 492
train_fraction = 0.7
fn_real = int(n_real * train_fraction)
fn_test = int(n_test * train_fraction)

real_samples = data.loc[data.Class == 0, test_cols].sample(n_real, replace=False).reset_index(drop=True)
test_samples = data.loc[data.Class == 1, test_cols].sample(n_test, replace=False).reset_index(drop=True)

train_df = pd.concat([real_samples[:fn_real], test_samples[:fn_test]], axis=0, ignore_index=True).reset_index(drop=True)
# train_df = pd.concat([real_samples[:fn_test],test_samples[:fn_test]],axis=0,ignore_index=True).reset_index(drop=True)

test_df = pd.concat([real_samples[fn_real:], test_samples[fn_test:]], axis=0, ignore_index=True).reset_index(drop=True)
print('classes 0, 1: ', n_real, n_test)
print('train, test: ', len(train_df), len(test_df))

X_col = test_df.columns[:-1]
y_col = test_df.columns[-1]
dtrain = xgb.DMatrix(train_df[X_col], train_df[y_col], feature_names=X_col)
dtest = xgb.DMatrix(test_df[X_col], test_df[y_col], feature_names=X_col)

results_dict = {}

xgb_params = {
    #     'max_depth': 4,
    'objective': 'binary:logistic',
    'random_state': 0,
    'eval_metric': 'auc',  # auc, error
    #     'tree_method': 'hist'
    #     'grow_policy': 'lossguide' # depthwise, lossguide
}

xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=100,
                     verbose_eval=False,
                     early_stopping_rounds=20,
                     evals=[(dtrain, 'train'), (dtest, 'test')],
                     evals_result=results_dict,
                     feval=recall, maximize=True
                     #                      feval = roc_auc, maximize=True
                     )

y_pred = xgb_test.predict(dtest, ntree_limit=xgb_test.best_iteration + 1)
y_true = test_df['Class'].values
print('best iteration: ', xgb_test.best_iteration)
print(recall(y_pred, dtest))
print(precision(y_pred, dtest))
print(roc_auc(y_pred, dtest))
# print( 'Accuracy: {:.3f}'.format(SimpleAccuracy(y_pred, y_true)) )
SimpleMetrics(np.round(y_pred), y_true)

for i in results_dict:
    for err in results_dict[i]:
        plt.plot(results_dict[i][err], label=i + ' ' + err)
plt.axvline(xgb_test.best_iteration, c='green', label='best iteration')
plt.xlabel('iteration')
# plt.ylabel(err)
plt.title('xgboost learning curves')
plt.legend()
plt.grid()

plt.savefig('plots/Plot7.png')
plt.show()
# Plot feature importances

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
xgb.plot_importance(xgb_test, max_num_features=10, height=0.5, ax=ax)
plt.savefig('plots/Plot8.png')

# Generate list of features sorted by importance in detecting fraud
# https://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value

import operator

x = xgb_test.get_fscore()
sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)

# print( 'Top eight features for fraud detection: ', [ i[0] for i in sorted_x[:8] ] )

sorted_cols = [i[0] for i in sorted_x] + ['Class']
print(sorted_cols)

# Plot all of the training data with paired features sorted by importance
# This takes a while

colors = ['red', 'blue']
markers = ['o', '^']
labels = ['real', 'fraud']
alphas = [0.7, 0.9]

columns = 4
rows = int(np.ceil(len(data_cols) / columns / 2))
plt.figure(figsize=(columns * 3.5, rows * 3))
plt.suptitle('XGBoost Sorted Data Distributions ', size=16)

train = train_df.copy()

for i in range(int(np.floor(len(sorted_x) / 2)))[:]:
    col1, col2 = sorted_x[i * 2][0], sorted_x[i * 2 + 1][0]
    #     print(i,col1,col2)

    plt.subplot(rows, columns, i + 1)
    for group, color, marker, label, alpha in zip(train.groupby('Class'), colors, markers, labels, alphas):
        plt.scatter(group[1][col1], group[1][col2],
                    label=label, marker=marker, alpha=alpha,
                    edgecolors=color, facecolors='none')
    plt.xlabel(col1, size=12)
    plt.ylabel(col2, size=12)
    plt.tick_params(axis='both', labelsize=10)
    if i == 0: plt.legend(fontsize=12, edgecolor='black')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # xmin, ymin, xmax, ymax
plt.savefig('plots/Plot9.png')
plt.show()

# Lets look at the effect of the ratio of normal:fraud data in the dataset on recall and roc_auc
# We'll use cross validation to see if differences are significant


np.random.seed(0)

n_real = np.sum(data.Class == 0)  # 200000
n_test = np.sum(data.Class == 1)  # 492
real_samples = data.loc[data.Class == 0, test_cols].sample(n_real, replace=False).reset_index(drop=True)
test_samples = data.loc[data.Class == 1, test_cols].sample(n_test, replace=False).reset_index(drop=True)
X_col = data.columns[:-1]
y_col = data.columns[-1]

test_data = []

# for i in [1]:
# for i in [0.1,0.5,1,2,10]:
for i in np.logspace(-1, 2, 8):
    print(i)
    train_df = pd.concat([real_samples[:int(n_test * i)], test_samples[:n_test]], axis=0,
                         ignore_index=True).reset_index(drop=True)
    dtrain = xgb.DMatrix(train_df[X_col], train_df[y_col], feature_names=X_col)
    results = xgb.cv(xgb_params, dtrain,
                     nfold=5, num_boost_round=100, early_stopping_rounds=10, seed=0,
                     feval=recall)
    test_data.append(list([i]) + list(results.tail(1).index) + list(results.tail(1).values[0]))

test_data = pd.DataFrame(test_data, columns=list(['ratio', 'best']) + list(results.columns))
test_data

# Recall decreases as more normal data is added

# metric = 'auc'
metric = 'recall'
# xs = test_data['ratio'].values
xs = np.log10(test_data['ratio'].values)
ys = test_data['test-' + metric + '-mean'].values
stds = test_data['test-' + metric + '-std'].values
plt.plot(xs, ys, c='C1')
plt.plot(xs, ys + stds, linestyle=':', c='C2')
plt.plot(xs, ys - stds, linestyle=':', c='C2')
plt.xlabel('log10 ratio of normal:fraud data')
plt.ylabel(metric)
# plt.ylim([0.96,1.01])

plt.savefig('plots/Plot10.png')
plt.show()
# load clustering libraries

import sklearn.cluster as cluster

# hdbscan not in kaggle/python at present

## !pip install hdbscan
import hdbscan

# Set up training set to consist of only fraud data

train = data.loc[data['Class'] == 1].copy()

print(pd.DataFrame([[np.sum(train['Class'] == i)] for i in np.unique(train['Class'])], columns=['count'],
                   index=np.unique(train['Class'])))

# train = pd.get_dummies(train, columns=['Class'], prefix='Class')
label_cols = [i for i in train.columns if 'Class' in i]
data_cols = [i for i in train.columns if i not in label_cols]
train_no_label = train[data_cols]

# TSNE is an interesting method to map higher dimensional data into two dimensions
# http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

# Note TSNE map may not be what you might think:
# https://distill.pub/2016/misread-tsne/

# Create multiple projections to compare results from different random states

from sklearn.manifold import TSNE

projections = [TSNE(random_state=i).fit_transform(train_no_label) for i in range(3)]

# Now we'll compare some different clustering algorithms
# https://github.com/scikit-learn-contrib/hdbscan/blob/master/docs/comparing_clustering_algorithms.rst

algorithms = [
    #     [ 'KMeans', cluster.KMeans, (), {'random_state':0} ],
    ['KMeans', cluster.KMeans, (), {'n_clusters': 2, 'random_state': 0}],
    #     [ 'KMeans 3', cluster.KMeans, (), {'n_clusters':3, 'random_state':0} ],
    #     [ 'Agglomerative', cluster.AgglomerativeClustering, (), {} ],
    ['Agglomerative', cluster.AgglomerativeClustering, (), {'linkage': 'ward', 'n_clusters': 3}],
    #     [ 'Agg. Ave 3', cluster.AgglomerativeClustering, (), {'linkage': 'average', 'n_clusters': 3} ],
    #     [ 'Agg. Complete 3', cluster.AgglomerativeClustering, (), {'linkage': 'complete', 'n_clusters': 3} ],
    #     [ 'DBSCAN', cluster.DBSCAN, (), {'eps':0.025} ],
    #     [ 'HDBSCAN', hdbscan.HDBSCAN, (), {} ],
    ['HDBSCAN', hdbscan.HDBSCAN, (), {'min_cluster_size': 10, 'min_samples': 1, }],
    #     [ 'HDBSCAN 2 10', hdbscan.HDBSCAN, (), {'min_cluster_size':2, 'min_samples':10, } ],
    #     [ 'HDBSCAN 10 10 ', hdbscan.HDBSCAN, (), {'min_cluster_size':10, 'min_samples':10, } ],
]

rows = len(algorithms)
columns = 4
plt.figure(figsize=(columns * 3, rows * 3))

for i, [name, algorithm, args, kwds] in enumerate(algorithms):
    print(i, name)

    labels = algorithm(*args, **kwds).fit_predict(train_no_label)
    #     labels = algorithm(*args, **kwds).fit_predict(projections[0])

    #     print( pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) ) )

    colors = np.clip(labels, -1, 9)
    colors = ['C' + str(i) if i > -1 else 'black' for i in colors]

    plt.subplot(rows, columns, i * columns + 1)
    plt.scatter(train_no_label[data_cols[0]], train_no_label[data_cols[1]], c=colors)
    plt.xlabel(data_cols[0]), plt.ylabel(data_cols[1])
    plt.title(name)

    for j in range(3):
        plt.subplot(rows, columns, i * columns + 1 + j + 1)
        plt.scatter(*(projections[j].T), c=colors)
        plt.xlabel('x'), plt.ylabel('y')
        plt.title('TSNE projection ' + str(j + 1), size=12)

#     break

plt.suptitle('Comparison of Fraud Clusters', size=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('plots/Plot11.png')
plt.show()
# Now pick a set of labels and add to the dataset

algorithm = cluster.KMeans
args, kwds = (), {'n_clusters': 2, 'random_state': 0}
labels = algorithm(*args, **kwds).fit_predict(train_no_label)
# labels = algorithm(*args, **kwds).fit_predict(projections[0])

print(pd.DataFrame([[np.sum(labels == i)] for i in np.unique(labels)], columns=['count'], index=np.unique(labels)))

fraud_w_classes = train.copy()
fraud_w_classes['Class'] = labels

# Let's see which features are most useful for detecting differences between the classes:

dtrain = xgb.DMatrix(fraud_w_classes[data_cols], fraud_w_classes['Class'], feature_names=data_cols)

xgb_params = {
    'objective': 'binary:logistic',
    'random_state': 0,
    'eval_metric': 'auc',  # allows for balanced or unbalanced classes
}
xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=10)  # limit to ten rounds for fast evaluation

import operator

x = xgb_test.get_fscore()
sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
print('Top eight features: ', [[i[0], i[1]] for i in sorted_x[:8]])


## GAN SETUP AND TRAINING

# reloading the libraries and setting the parameters


import GAN1
import importlib

importlib.reload(GAN1)  # For reloading after making changes
from GAN1 import *

rand_dim = 32  # 32 # needs to be ~data_dim
base_n_count = 128  # 128

nb_steps = 500 + 1  # 50000 # Add one for logging of the last interval
batch_size = 128  # 64

k_d = 1  # number of critic network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
critic_pre_train_steps = 100  # 100  # number of steps to pre-train the critic before starting adversarial training
log_interval = 100  # 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
learning_rate = 5e-4  # 5e-5
data_dir = 'cache/'
generator_model_path, discriminator_model_path, loss_pickle_path = None, None, None

# show = False
show = True

# train = create_toy_spiral_df(1000)
# train = create_toy_df(n=1000,n_dim=2,n_classes=4,seed=0)
train = fraud_w_classes.copy().reset_index(drop=True)  # fraud only with labels from classification

# train = pd.get_dummies(train, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [i for i in train.columns if 'Class' in i]
data_cols = [i for i in train.columns if i not in label_cols]
train[data_cols] = train[data_cols] / 10  # scale to random noise size, one less thing to learn
train_no_label = train[data_cols]

# Training the vanilla GAN and CGAN architectures

k_d = 1  # number of critic network updates per adversarial training step
learning_rate = 5e-4  # 5e-5
arguments = [rand_dim, nb_steps, batch_size,
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
             data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show]

adversarial_training_GAN(arguments, train_no_label, data_cols)  # GAN
#uncommented next line
adversarial_training_GAN(arguments, train, data_cols=data_cols, label_cols=label_cols ) # CGAN

# Training the WGAN and WCGAN architectures

k_d = 5  # train critic to optimal state each time
learning_rate = 1e-4  # 5e-5
arguments = [rand_dim, nb_steps, batch_size,
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
             data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show]

adversarial_training_WGAN(arguments, train_no_label, data_cols=data_cols)  # WGAN
adversarial_training_WGAN(arguments, train, data_cols=data_cols, label_cols=label_cols)  # WCGAN

# %%time

# # for continued training

# import GAN_1711103
# import importlib
# importlib.reload(GAN_171103) # For reloading after making changes
# from GAN_171103 import *

# last_step = 1000
# prefix = 'WGAN'
# # data_dir = 'cache lr mix base 128 act mix 171026/'
# data_dir = 'cache/'

# # Choose your learning rate
# # learning_rate = 1e-5 # first 10k
# # learning_rate = 1e-5 # 10-15k
# # learning_rate = 1e-6 # 15-20k

# generator_model_path = data_dir + prefix + '_generator_model_weights_step_' + str(last_step) + '.h5'
# discriminator_model_path = data_dir + prefix + '_discriminator_model_weights_step_' + str(last_step) + '.h5'
# loss_pickle_path = data_dir + prefix + '_losses_step_' + str(last_step) + '.pkl'

# nb_steps = 4000

# arguments = [rand_dim, nb_steps, batch_size,
#              k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
#             data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ]

# # Choose your training algorithm
# # adversarial_training_GAN(arguments, train_no_label, data_cols=data_cols, starting_step=last_step+1 ) # GAN
# # adversarial_training_GAN(arguments, train, data_cols=data_cols, label_cols=label_cols, starting_step=last_step+1 ) # CGAN
# # adversarial_training_WGAN(arguments, train_no_label, data_cols=data_cols, starting_step=last_step+1 ) # WGAN
# adversarial_training_WGAN(arguments, train, data_cols=data_cols, label_cols=label_cols, starting_step=last_step+1 ) # WCGAN

# For reloading loss data from pickles

prefix = 'WCGAN'
step = 500

[combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(
    open(data_dir + prefix + '_losses_step_' + str(step) + '.pkl', 'rb'))

# plt.plot( xgb_losses[:] ) ;
w = 10
plt.plot(pd.DataFrame(xgb_losses[:]).rolling(w).mean())

plt.savefig('plots/Plot36.png')
plt.show()

best_step = list(xgb_losses).index(xgb_losses.min()) * 10
print(best_step, xgb_losses.min())

xgb100 = [xgb_losses[i] for i in range(0, len(xgb_losses), 10)]
best_step = xgb100.index(min(xgb100)) * log_interval
print(best_step, min(xgb100))

# Look for the step with the lowest critic loss, and the lowest step saved (every 100)

delta_losses = np.array(disc_loss_real) - np.array(disc_loss_generated)

best_step = list(delta_losses).index(delta_losses.min())
print(best_step, delta_losses.min())

delta100 = [delta_losses[i] for i in range(0, len(delta_losses), 100)]
best_step = delta100.index(min(delta100)) * log_interval
print(best_step, min(delta100))

plt.plot( (np.array(disc_loss_real) - np.array(disc_loss_generated)) ) #uncommented

w = 50
# plt.plot( list(range(0,5001,1)), pd.rolling_mean((np.array(disc_loss_real) - np.array(disc_loss_generated)),w) )
plt.plot(pd.DataFrame(disc_loss_real[:]).rolling(w).mean() - pd.DataFrame(disc_loss_generated[:]).rolling(w).mean());

plt.xlim([9000,10000])
plt.ylim([0.03,0.05])

plt.savefig('plots/Plots37.png')
plt.show()
# Let's look at some of the generated data
# First create the networks locally and load the weights

import GAN1
import importlib

importlib.reload(GAN1)  # For reloading after making changes
from GAN1 import *

seed = 17

train = fraud_w_classes.copy().reset_index(drop=True)  # fraud only with labels from classification

# train = pd.get_dummies(train, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [i for i in train.columns if 'Class' in i]
data_cols = [i for i in train.columns if i not in label_cols]
train[data_cols] = train[data_cols] / 10  # scale to random noise size, one less thing to learn
train_no_label = train[data_cols]

data_dim = len(data_cols)
label_dim = len(label_cols)
with_class = False
if label_dim > 0: with_class = True
np.random.seed(seed)

# define network models

generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type='Wasserstein')
#generator_model.load_weights('cache/WCGAN_generator_model_weights_step_4800.h5')
generator_model.load_weights('cache/WCGAN_generator_model_weights_step_400.h5')
#uncommented next 2 lines
generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count)
generator_model.load_weights('cache/CGAN_generator_model_weights_step_500.h5')

# with_class = False
# train = train_no_label
# label_cols = []
# # generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim, base_n_count, type='Wasserstein')
# # generator_model.load_weights('cache/WGAN_generator_model_weights_step_4800.h5')

# generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim, base_n_count)
# generator_model.load_weights('cache/GAN_generator_model_weights_step_5000.h5')

# Now generate some new data

test_size = 492  # Equal to all of the fraud cases

x = get_data_batch(train, test_size, seed=i + j)
z = np.random.normal(size=(test_size, rand_dim))
if with_class:
    labels = x[:, -label_dim:]
    g_z = generator_model.predict([z, labels])
else:
    g_z = generator_model.predict(z)

# Check using the same functions used during GAN training

print(CheckAccuracy(x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim))

PlotData(x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim)

# Now we can train and test an xgboost classifier on our generated data

real_samples = pd.DataFrame(x, columns=data_cols + label_cols)
test_samples = pd.DataFrame(g_z, columns=data_cols + label_cols)
real_samples['syn_label'] = 0
test_samples['syn_label'] = 1

training_fraction = 0.5
n_real, n_test = int(len(real_samples) * training_fraction), int(len(test_samples) * training_fraction)
train_df = pd.concat([real_samples[:n_real], test_samples[:n_test]], axis=0)
test_df = pd.concat([real_samples[n_real:], test_samples[n_test:]], axis=0)

# X_col = test_df.columns[:-(label_dim+1)]
X_col = test_df.columns[:-1]
y_col = test_df.columns[-1]
dtrain = xgb.DMatrix(train_df[X_col], train_df[y_col], feature_names=X_col)
dtest = xgb.DMatrix(test_df[X_col], feature_names=X_col)
y_true = test_df['syn_label']

# dtrain = np.vstack( [ x[:int(len(x)/2)], g_z[:int(len(g_z)/2)] ] )
# dlabels = np.hstack( [ np.zeros(int(len(x)/2)), np.ones(int(len(g_z)/2)) ] )
# dtest = np.vstack( [ x[int(len(x)/2):], g_z[int(len(g_z)/2):] ] )

# dtrain = xgb.DMatrix(dtrain, dlabels, feature_names=data_cols+label_cols)
# dtest = xgb.DMatrix(dtest, feature_names=data_cols+label_cols)
# y_true = dlabels

xgb_params = {
    'max_depth': 4,
    'objective': 'binary:logistic',
    'random_state': 0,
    'eval_metric': 'auc',  # allows for balanced or unbalanced classes
}
xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=10)

y_pred = np.round(xgb_test.predict(dtest))

print('{:.2f}'.format(SimpleAccuracy(y_pred, y_true)))

# Let's look at how the discrimnator scored real and generated data, visualized along every feature

y_pred0 = xgb_test.predict(dtest)

for i in range(0, len(X_col) - 1, 2):

    f, axarr = plt.subplots(1, 2, figsize=(6, 2))

    axarr[0].scatter(test_df[:n_real][X_col[i]], test_df[:n_real][X_col[i + 1]], c=y_pred0[:n_real], cmap='plasma')
    axarr[0].set_title('real')
    axarr[0].set_ylabel(X_col[i + 1])

    axarr[1].scatter(test_df[n_real:][X_col[i]], test_df[n_real:][X_col[i + 1]], c=y_pred0[n_real:], cmap='plasma')
    axarr[1].set_title('test')
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim())

    for a in axarr:
        a.set_xlabel(X_col[i])

    plt.show()

colors = ['red', 'blue']
markers = ['o', '^']
labels = ['real', 'fraud']

class_label = 'Class'

for i in range(0, len(X_col), 2):
    col1, col2 = i, i + 1
    if i + 1 >= len(X_col): continue

    f, axarr = plt.subplots(1, 2, figsize=(6, 2))
    for group, color, marker, label in zip(test_df[:n_real].groupby(class_label), colors, markers, labels):
        axarr[0].scatter(group[1][X_col[col1]], group[1][X_col[col2]], label=label, c=color, marker=marker, alpha=0.2)
    axarr[0].legend()
    axarr[0].set_title('real')
    axarr[0].set_ylabel(X_col[col2])

    for group, color, marker, label in zip(test_df[n_real:].groupby(class_label), colors, markers, labels):
        axarr[1].scatter(group[1][X_col[col1]], group[1][X_col[col2]], label=label, c=color, marker=marker, alpha=0.2)
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim())
    axarr[1].legend()
    axarr[1].set_title('generated');

    for a in axarr:
        a.set_xlabel(X_col[col1])

    plt.show()

# Evaluate performance on validation set

SimpleMetrics(y_pred, y_true)

# Plot feature importances used for identifying generated data

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
xgb.plot_importance(xgb_test, max_num_features=20, height=0.5, ax=ax)
plt.savefig('plots/Plots68.png')

## COMPARE GAN OUTPUT ##

# Set up the training dataset
train = fraud_w_classes.copy().reset_index(drop=True) # fraud only with labels from classification

train = pd.get_dummies(train, columns=['Class'], prefix='Class', drop_first=True)
label_cols = [ i for i in train.columns if 'Class' in i ]
data_cols = [ i for i in train.columns if i not in label_cols ]
train[ data_cols ] = train[ data_cols ] / 10 # scale to random noise size, one less thing to learn
train_no_label = train[ data_cols ]

data_dim = len(data_cols)
label_dim = len(label_cols)

# Generate empty models

rand_dim = 32
base_n_count = 128
model_names = ['GAN', 'CGAN', 'WGAN', 'WCGAN']
with_classes = [False, True, False, True]
type0s = [None, None, 'Wasserstein', 'Wasserstein']

models = {}

for model_name, with_class, type0 in zip(model_names, with_classes, type0s):

    if with_class:
        generator_model, discriminator_model, combined_model = \
            define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type=type0)
    else:
        generator_model, discriminator_model, combined_model = \
            define_models_GAN(rand_dim, data_dim, base_n_count, type=type0)

    models[model_name] = [model_name, with_class, type0, generator_model]

# Setup parameters

seed = 17
test_size = 492 # number of fraud cases

np.random.seed(seed)
z = np.random.normal(size=(test_size, rand_dim))
x = get_data_batch(train, test_size, seed=seed)
real_samples = pd.DataFrame(x, columns=data_cols+label_cols)
labels = x[:,-label_dim:]

# colors = ['C1','C9']
# colors = ['xkcd:plum', 'xkcd:navy']
colors = ['red','blue']
markers = ['o','^']
class_labels = ['Class 1','Class 2']

col1, col2 = 'V17', 'V10'

#base_dir = 'cache lr mix base 128 act mix 171026/'
base_dir = 'cache/'

## COMPARISON OF GAN OUTPUTS ##

# model_steps = [500, 5000]
# model_steps = [ 0, 100, 200, 500, 1000 ]
#model_steps = [0, 100, 200, 500, 1000, 2000, 5000]
model_steps = [0, 100, 200, 300, 400, 500]
rows = len(model_steps)
columns = 5

axarr = [[]] * len(model_steps)

fig = plt.figure(figsize=(14, rows * 3))

for model_step_ix, model_step in enumerate(model_steps):
    print(model_step)

    axarr[model_step_ix] = plt.subplot(rows, columns, model_step_ix * columns + 1)

    for group, color, marker, label in zip(real_samples.groupby('Class_1'), colors, markers, class_labels):
        plt.scatter(group[1][[col1]], group[1][[col2]],
                    label=label, marker=marker, edgecolors=color, facecolors='none')

    plt.title('Actual Fraud Data')
    plt.ylabel(col2)  # Only add y label to left plot
    plt.xlabel(col1)
    xlims, ylims = axarr[model_step_ix].get_xlim(), axarr[model_step_ix].get_ylim()

    if model_step_ix == 0:
        legend = plt.legend()
        legend.get_frame().set_facecolor('white')

    for i, model_name in enumerate(model_names[:]):

        [model_name, with_class, type0, generator_model] = models[model_name]

        generator_model.load_weights(base_dir + model_name + '_generator_model_weights_step_' + str(model_step) + '.h5')

        ax = plt.subplot(rows, columns, model_step_ix * columns + 1 + (i + 1))

        if with_class:
            g_z = generator_model.predict([z, labels])
            gen_samples = pd.DataFrame(g_z, columns=data_cols + label_cols)
            for group, color, marker, label in zip(gen_samples.groupby('Class_1'), colors, markers, class_labels):
                plt.scatter(group[1][[col1]], group[1][[col2]],
                            label=label, marker=marker, edgecolors=color, facecolors='none')
        else:
            g_z = generator_model.predict(z)
            gen_samples = pd.DataFrame(g_z, columns=data_cols)
            plt.scatter(gen_samples[[col1]], gen_samples[[col2]],
                        label=class_labels[0], marker=markers[0], edgecolors=colors[0], facecolors='none')
        plt.title(model_name)
        plt.xlabel(data_cols[0])
        ax.set_xlim(xlims), ax.set_ylim(ylims)

plt.suptitle('Comparison of GAN outputs', size=16)
plt.tight_layout(rect=[0.075, 0, 1, 0.95])


# Adding text labels for traning steps
vpositions = np.array([i._position.bounds[1] for i in axarr])
vpositions += ((vpositions[0] - vpositions[1]) * 0.35)
for model_step_ix, model_step in enumerate(model_steps):
    fig.text(0.05, vpositions[model_step_ix], 'training\nstep\n' + str(model_step), ha='center', va='center', size=12)

plt.savefig('plots/Plots69.png')

##GENERATED DATA TESTING ##
### THIS SECTION TAKES ABOUT 20 MINUTES TO RUN ###

# Setup xgboost parameters

xgb_params = {
#     'max_depth': 4,
    'objective': 'binary:logistic',
    'random_state': 0,
    'eval_metric': 'auc', # auc, error
#     'tree_method': 'hist'
#     'grow_policy': 'lossguide' # depthwise, lossguide
}

# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py

from sklearn.metrics import recall_score, precision_score, roc_auc_score

def recall(preds, dtrain):
    labels = dtrain.get_label()
    return 'recall',  recall_score(labels, np.round(preds))

def precision(preds, dtrain):
    labels = dtrain.get_label()
    return 'precision',  precision_score(labels, np.round(preds))

def roc_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'roc_auc',  roc_auc_score(labels, preds)

# Define model parameters

seed = 17
np.random.seed(seed)

data_dim = len(data_cols)
label_dim = len(label_cols)

base_dir = 'cache/'
#base_dir = 'cache lr mix base 128 act mix 171026/'
rand_dim = 32
base_n_count = 128

# defined training set parameters

train_fraction = 0.7
X_col = data.columns[:-1]
y_col = data.columns[-1]

folds = 5


# Function to make cross folds with different amounts of an additional dataset added

def MakeCrossFolds(g_z_df=[]):
    np.random.seed(0)

    train_real_set, test_real_set = [], []
    train_fraud_set, test_fraud_set = [], []

    real_samples = data.loc[data.Class == 0].copy()
    fraud_samples = data.loc[data.Class == 1].copy()

    #     n_temp_real = 10000
    n_temp_real = len(real_samples)

    for seed in range(folds):
        np.random.seed(seed)

        fraud_samples = fraud_samples.sample(len(fraud_samples), replace=False).reset_index(drop=True)  # shuffle

        #     n_train_fraud = int(len(fraud_samples) * train_fraction)
        n_train_fraud = 100
        train_fraud_samples = fraud_samples[:n_train_fraud].reset_index(drop=True)

        #     test_fraud_samples = fraud_samples[n_train_fraud:].reset_index(drop=True)
        n_test_fraud = 148  # 30% left out
        test_fraud_samples = fraud_samples[-n_test_fraud:].reset_index(drop=True)

        if len(g_z_df) == 0: g_z_df = fraud_samples[
                                      n_train_fraud:-n_test_fraud]  # for adding real data, if no generated
        n_g_z = len(g_z_df)
        train_fraud_samples = train_fraud_samples.append(g_z_df).reset_index(drop=True)

        real_samples = real_samples.sample(len(real_samples), replace=False).reset_index(drop=True)  # shuffle
        temp_real_samples = real_samples[:n_temp_real]
        n_train_real = int(len(temp_real_samples) * train_fraction)

        train_real_samples = temp_real_samples[:n_train_real].reset_index(drop=True)  # with margin
        test_real_samples = temp_real_samples[n_train_real:].reset_index(drop=True)  # with margin

        train_real_set.append(train_real_samples)
        test_real_set.append(test_real_samples)
        train_fraud_set.append(train_fraud_samples)
        test_fraud_set.append(test_fraud_samples)

    print(n_train_fraud)
    for i in [fraud_samples, g_z_df, train_fraud_samples, test_fraud_samples]: print(len(i))
    for i in [real_samples, train_real_samples, test_real_samples]: print(len(i))
    # [ [ len(i) for i in j ] for j in [train_real_set, test_real_set, train_fraud_set, test_fraud_set] ]

    return n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set


def Run_CV_Xgb(n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set):

    test_data=[]

    # for i in [1]:
    # for i in [1,2,5,10,20]:
    # for i in np.logspace(0,np.log10(11),num=5):
    # for i in np.logspace(0,np.log10(11),num=3):
    for i in np.logspace(0,np.log10((492-148)/100),num=5):

        print('# additional generated data tested: {}'.format (int(n_train_fraud*(i-1)) ) )
        for k in range(folds):

            train_df = pd.concat(
                [ train_real_set[k], train_fraud_set[k][:int(n_train_fraud*i)] ],
                 axis=0,ignore_index=True).reset_index(drop=True)

            test_df = pd.concat(
                [ test_real_set[k], test_fraud_set[k] ],
                axis=0,ignore_index=True).reset_index(drop=True)

            dtrain = xgb.DMatrix(train_df[X_col], train_df[y_col], feature_names=X_col)
            dtest = xgb.DMatrix(test_df[X_col], test_df[y_col], feature_names=X_col)

            results_dict = {}
            xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=100,
                                 verbose_eval=False, early_stopping_rounds=10,
                                 evals=[(dtrain,'train'),(dtest,'test')],
                                 evals_result = results_dict )

            y_pred = xgb_test.predict(dtest, ntree_limit=xgb_test.best_iteration+1)
            y_true = test_df['Class'].values
            results = [k, i, xgb_test.best_iteration, recall( y_pred, dtest )[1], precision( y_pred, dtest )[1], roc_auc( y_pred, dtest )[1] ]
    #         print(results)

            test_data.append(results)
    test_data = pd.DataFrame(test_data, columns=['k', 'ratio','best','recall','precision','auc'])
    return test_data


# Generate and test data with untrained model

os.system('time')
generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count,type='Wasserstein')
generator_model.load_weights(base_dir + 'WCGAN_generator_model_weights_step_0.h5')

test_size = 492
x = get_data_batch(fraud_w_classes, test_size, seed=0)
z = np.random.normal(size=(test_size, rand_dim))
labels = x[:, -label_dim:]
g_z = generator_model.predict([z, labels])

# The labels for the generate data will all be 1, as they are supposed to be fraud data
g_z_df = pd.DataFrame(np.hstack([g_z[:, :len(data_cols)], np.ones((len(g_z), 1))]), columns=data.columns)

n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set = MakeCrossFolds(g_z_df)

t_0 = Run_CV_Xgb(n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set)

# Generate and test data with trained model

os.system('time')

generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count,type='Wasserstein')
generator_model.load_weights(base_dir + 'WCGAN_generator_model_weights_step_400.h5')

test_size = 492
x = get_data_batch(fraud_w_classes, test_size, seed=0)
z = np.random.normal(size=(test_size, rand_dim))
labels = x[:, -label_dim:]
g_z = generator_model.predict([z, labels])

# The labels for the generate data will all be 1, as they are supposed to be fraud data
g_z_df = pd.DataFrame(np.hstack([g_z[:, :len(data_cols)], np.ones((len(g_z), 1))]), columns=data.columns)

n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set = MakeCrossFolds(g_z_df)

t_4800 = Run_CV_Xgb(n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set)

# Generate and test data with additional real data

os.system('time')

n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set = MakeCrossFolds()

t_real = Run_CV_Xgb(n_train_fraud, train_real_set, test_real_set, train_fraud_set, test_fraud_set)

# # Save the testing data

# # Run using the WCGAN trained after 0 steps
pickle.dump( t_0, open('cache/additional untrained generated fraud data test.pkl','wb'))

# # Run using the WCGAN trained after 4800 steps
pickle.dump( t_4800, open('cache/additional generated fraud data test.pkl','wb'))

# # Run using the real data
pickle.dump( t_real, open('cache/additional real fraud data test.pkl','wb'))

# Reload the testing data

t_0 = pickle.load(open('cache/additional untrained generated fraud data test.pkl','rb'))
t_4800 = pickle.load(open('cache/additional generated fraud data test.pkl','rb'))
t_real = pickle.load(open('cache/additional real fraud data test.pkl','rb'))

##EFFECTS OF ADDITIONAL DATA ON FRAUD DETERCTION ##

# Plot the testing data

labels = ['WCGAN\ntrained 0 steps','WCGAN\ntrained 4800 steps','Actual Fraud Data']

metric = 'recall'

plt.figure(figsize=(12,3))
for i, [label, test_data] in enumerate(zip(labels, [t_0, t_4800, t_real])):

    xs = [ n_train_fraud * (i[0]-1) for i in test_data.groupby('ratio') ]
    ys = test_data.groupby('ratio')[metric].mean().values
    stds = 2 * test_data.groupby('ratio')[metric].std().values

    plt.subplot(1,3,i+1)
    plt.axhline(ys[0],linestyle='--',color='red')
    plt.plot(xs,ys,c='C1',marker='o')
    plt.plot(xs,ys+stds,linestyle=':',c='C2')
    plt.plot(xs,ys-stds,linestyle=':',c='C2')
    if i==0: plt.ylabel(metric)
    plt.xlabel('# additional data')
    plt.title(label,size=12)
    plt.xlim([0,11])
    plt.ylim([0.55,.85])
    plt.ylim([0.6,1.0])

plt.tight_layout(rect=[0,0,1,0.9])
plt.suptitle('Effects of additional data on fraud detection', size=16)
plt.savefig('plots/Plot70.png')
plt.show()

##SUMMARY OF TRAINING DATA ##

# Load the saved loss data from each model

base_dir = 'cache/'
#base_dir = 'cache lr mix base 128 act mix 171026/'

suffix = '_step_500'

GAN_losses = pickle.load(open(base_dir + 'GAN_losses'+suffix+'.pkl','rb'))
# GAN_losses = [combined_loss, disc_loss_real, disc_loss_generated, xgb_losses]

CGAN_losses = pickle.load(open(base_dir + 'CGAN_losses'+suffix+'.pkl','rb'))
WGAN_losses = pickle.load(open(base_dir + 'WGAN_losses'+suffix+'.pkl','rb'))
WCGAN_losses = pickle.load(open(base_dir + 'WCGAN_losses'+suffix+'.pkl','rb'))

# Find best xgb scores overall and saved (every 100 steps)

data_ix = 3
data_sets = [ GAN_losses[data_ix], CGAN_losses[data_ix], WGAN_losses[data_ix], WCGAN_losses[data_ix]]
labels = [ 'GAN','CGAN','WGAN','WCGAN' ]

for label, data_set in zip( labels, data_sets ):
    best_step = list(data_set).index( np.array(data_set).min() ) * 10
    print( '{: <5} step {: <4}: {:.4f}'.format( label, best_step, np.array(data_set).min() ) )

    xgb100 = [ data_set[i] for i in range(0, len(data_set), 10) ]
    best_step = xgb100.index( min(xgb100) ) * 100
    print( '{: <5} step {: <4}: {:.4f}\n'.format( label, best_step, np.array(xgb100).min() ) )
    print( best_step, min(xgb100) )

# Look at the unsmoothed losses

data_fields = ['combined_losses_', 'real_losses_', 'generated_losses_', 'xgb_losses']
sampling_intervals = [ 1, 1, 1, 10 ]
labels = [ 'GAN','CGAN','WGAN','WCGAN' ]
linestyles = ['-', '--', '-.', ':']

for data_ix in range(len(data_fields)):
    data_sets = [ GAN_losses[data_ix], CGAN_losses[data_ix], WGAN_losses[data_ix], WCGAN_losses[data_ix]]

    plt.figure(figsize=(10,5))
    for data, label, linestyle in zip(data_sets, labels, linestyles):
        plt.plot( np.array(range(0,len(data)))*sampling_intervals[data_ix],
                 data,
                 label=label, linestyle=linestyle )

    plt.ylabel(data_fields[data_ix])
    plt.xlabel('training step')
    plt.legend()
    plt.show()

# Look at the smoothed losses

data_fields = ['combined_losses_', 'real_losses_', 'generated_losses_', 'xgb_losses']
sampling_intervals = [ 1, 1, 1, 10 ]
labels = [ 'GAN','CGAN','WGAN','WCGAN' ]
linestyles = ['-', '--', '-.', ':']

w = 100
for data_ix in range(len(data_fields)):
    data_sets = [ GAN_losses[data_ix], CGAN_losses[data_ix], WGAN_losses[data_ix], WCGAN_losses[data_ix]]

    plt.figure(figsize=(10,5))
    for data, label, linestyle in zip(data_sets, labels, linestyles):
        plt.plot( np.array(range(0,len(data)))*sampling_intervals[data_ix],
                 pd.DataFrame(data).rolling(w).mean(),
                 label=label, linestyle=linestyle )

    plt.ylabel(data_fields[data_ix])
    plt.xlabel('training step')
    plt.legend()
    plt.show()

##ACCURACY OF GENERATED DATA DETECTION ##

# Create a figure for the smoothed xgboost losses

data_fields = ['combined_losses_', 'real_losses_', 'generated_losses_', 'xgb_losses']
sampling_intervals = [ 1, 1, 1, 10 ]
labels = [ 'GAN','CGAN','WGAN','WCGAN' ]
linestyles = ['-', '--', '-.', ':']

w = 50
data_ix = 3

data_sets = [ GAN_losses[data_ix], CGAN_losses[data_ix], WGAN_losses[data_ix], WCGAN_losses[data_ix]]

plt.figure(figsize=(10,5))
for data, label, linestyle in zip(data_sets, labels, linestyles):
    plt.plot( np.array(range(0,len(data)))*sampling_intervals[data_ix],
             pd.DataFrame(data).rolling(w).mean(),
             label=label, linestyle=linestyle )

plt.ylabel(data_fields[data_ix])
plt.xlabel('training step')
legend = plt.legend()
legend.get_frame().set_facecolor('white')

plt.title('Accuracy of generated data detection')
plt.ylabel('xgboost accuracy')
plt.tight_layout() ;
plt.savefig('plots/Plot79.png')

## DIFFERENCES IN CRITICAL DATA LOSS ##

# Create a figure for the critic losses for the WGAN and WCGAN

w = 50

data_ix0 = 2
data_ix1 = 1
data_fields = ['combined_losses_', 'real_losses_', 'generated_losses_', 'xgb_losses']

i1, i2 = 2, 3
i2 += 1

labels = [ 'GAN','CGAN','WGAN','WCGAN' ][i1:i2]
data_sets0 = [ GAN_losses[data_ix0], CGAN_losses[data_ix0], WGAN_losses[data_ix0], WCGAN_losses[data_ix0]][i1:i2]
data_sets1 = [ GAN_losses[data_ix1], CGAN_losses[data_ix1], WGAN_losses[data_ix1], WCGAN_losses[data_ix1]][i1:i2]
linestyles = ['-', '--', '-.', ':'][i1:i2]

plt.figure(figsize=(10,5))
for data0, data1, label, linestyle in zip(data_sets0, data_sets1, labels, linestyles):
    plt.plot( range(0,len(data0)),
             pd.DataFrame( np.array(data0)-np.array(data1) ).rolling(w).mean(),
             label=label, linestyle=linestyle )
plt.title('Difference between critic loss (EM distance estimate)\non generated samples and real samples')
plt.xlabel('training step')
plt.ylabel('Gen - Real Critic Loss')
legend = plt.legend()
legend.get_frame().set_facecolor('white')

plt.savefig('plots/Plot80.png')
plt.show()
