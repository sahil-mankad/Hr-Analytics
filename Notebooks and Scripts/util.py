"""
Commonly used utility functions
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, chi2
from scipy import stats
from statsmodels.stats import weightstats as stests
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, RandomizedSearchCV, train_test_split, KFold
from statistics import mean
from collections import Counter
from imblearn.combine import SMOTEENN
import random
from sklearn.naive_bayes import ComplementNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

color = sns.color_palette()
sns.set()
sns.set(style="darkgrid")

# For getting data types of dataframes
def get_dtypes(df):
  num_cols = list(df.select_dtypes(exclude = 'object').columns)
  cat_cols = [column for column in df.columns if column not in num_cols]
  
  return num_cols,cat_cols

def convert_to_categorical(df,col):
  return df[col].apply(lambda x:str(x))

def plot_numerical(df_col):
  fig, axes = plt.subplots(nrows =1 , ncols=2, figsize=(12, 6))
  fig.tight_layout()
  ax1,ax2 = axes[0] , axes[1]  
  sns.boxplot(df_col, orient='vertical',ax=ax1)
  sns.distplot(df_col, kde = False, bins = 20, ax = ax2)
  plt.show()

def plot_categorical(df,col):
  print("Column name: ",col)
  print(df[col].value_counts(normalize=True) * 100)
  sns.countplot(df[col])
  plt.show()

def barplot_percentages(df,feature,target):
    ax1 = df.groupby(feature)[target].value_counts(normalize=True).unstack() * 100
    ax1.plot(kind='bar', stacked=True)
    int_level = df[feature].value_counts()

def chi_square(df,column,target):
  table = pd.crosstab(df[column],df[target],margins=True)
  table.drop('All',axis=1,inplace=True)
  table.drop('All',axis=0,inplace=True)
  stat, p, dof, expected = chi2_contingency(table)
  print('dof=%d' % dof)
  print("Expected \n",expected)
  print("Observed \n",table)

  # interpret test-statistic
  prob = 0.95
  critical = chi2.ppf(prob, dof)
  print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
  if abs(stat) >= critical:
    print('Dependent (reject Null Hypothesis)')
  else:
    print('Independent (fail to reject Null Hypothesis)')
  # interpret p-value
  alpha = 1.0 - prob
  print('significance=%.3f, p=%.3f' % (alpha, p))
  if p <= alpha:
    print('Dependent (reject Null Hypothesis)')
  else:
    print('Independent (fail to reject Null Hypothesis)')

#cat_col: name of categorical variable, cont_col:name of continuous variable
def cont_to_cat(df,cat_col,cont_col):
  temp = []
  no_rows = df.shape[0]
  for i in range(0, no_rows):
    if int(df[cont_col][i]) > 1:
      temp.append(1)
    else:
      temp.append(0)
  
  df[cat_col] = pd.Series(temp).astype('object')
  print(df[cat_col].value_counts())
  return df

def get_unique_instances(df,cols):
  for i in range(0,len(cols)):
    print(str(cols[i]) + " - Number of Unique Values: " + str(df[cols[i]].nunique()))

def one_hot_encoding(data,cat_feats):
  one_hot = pd.get_dummies(data[cat_feats])
  one_hot = one_hot.astype('object')
  data.drop(cat_feats,axis=1,inplace=True)
  data = pd.concat([data,one_hot],axis=1)
  return data

def min_max_normalization(df):
  return (df - df.min())/(df.max()-df.min())

def check_clf(X,y,metric):
  clf_models = [ComplementNB(),LGBMClassifier(),CatBoostClassifier(verbose=0),LogisticRegression(class_weight='balanced'),DecisionTreeClassifier(class_weight='balanced'),SVC(class_weight='balanced'),
                RandomForestClassifier(class_weight='balanced'),AdaBoostClassifier(),XGBClassifier(scale_pos_weight=91.5,random_state=96)]
  
  clf_names = ['ComplementNB','LGBMClassifier','CatBoostClassifier','LogisticRegression','DecisionTreeClassifier','SVC','RandomForestClassifier','AdaBoostClassifier','XGBClassifier']
  
  results = []
  for clf in clf_models:
    #Using a numeric value for cv means that we'll be useing stratified k-fold cross validation
    scores = cross_val_score(clf, X, y, cv = 5, scoring=metric,n_jobs=-1)
    results.append(mean(scores))
  
  result_dict = {}
  
  for i in range(len(results)):
    print("Name: ",clf_names[i])
    print("Score: ",results[i])
    result_dict[clf_names[i]] = results[i] 

  return result_dict


#Check performance of model
def check_clf_score(clf, X_train, y_train, X_valid, y_valid, X_test,y_test):
  clf.fit(X_train,y_train)
  y_train_pred = clf.predict(X_train)
  y_valid_pred =  clf.predict(X_valid)
  train_score = f1_score(y_train, y_train_pred, average='macro')
  valid_score = f1_score(y_valid, y_valid_pred, average='macro')

  print("Training score: ",train_score)
  print("valid score: ",valid_score)

  y_test_pred =  clf.predict(X_test)
  test_score = f1_score(y_test, y_test_pred, average='macro')
  print("test score: ",test_score)

  print("\n For train set \n")
  print("Actual value counts")
  print(pd.Series(y_train).value_counts())
  print("Predicted value counts")
  print(pd.Series(y_train_pred).value_counts())

  print("\n For validation set \n")
  print("Actual value counts")
  print(pd.Series(y_valid).value_counts())
  print("Predicted value counts")
  print(pd.Series(y_valid_pred).value_counts())
  
  print("\n For test set \n")
  print("Actual value counts")
  print(pd.Series(y_test).value_counts())
  print("Predicted value counts")
  print(pd.Series(y_test_pred).value_counts())
  return y_train_pred,y_valid_pred,y_test_pred

def get_optimal_f1_thresh(target, oofs):
  thresholds = np.arange(0, 100)/100
  thresh_scores = []
  for thresh in thresholds:
    oofs_rounded = (oofs > thresh) * 1
    thresh_score = f1_score(target, oofs_rounded,average='macro')
    thresh_scores.append(thresh_score)
  
  all_thresholds_and_scores = pd.Series(index = thresholds, data = thresh_scores)
  all_thresholds_and_scores.plot(figsize=(10, 6), fontsize=14)
  
  plt.xlabel('Threshold', fontsize=14)
  plt.ylabel('F1 Score', fontsize=14)
  print("Best Threshold: ",all_thresholds_and_scores.sort_values(ascending=False).index.values[0])
  print("Best Score: ",all_thresholds_and_scores.sort_values(ascending=False)[0])

def random_search(clf,params,model_name,num_iter,X_train,y_train):
  randomsearch = RandomizedSearchCV(estimator=clf, param_distributions=params, n_iter= num_iter, cv=5, scoring = 'f1_macro',n_jobs = -1)
  randomsearch.fit(X_train,y_train)
  best_params = randomsearch.best_params_
  filename = model_name+'_best_params.sav'
  pickle.dump(best_params, open(filename, 'wb'))
  return best_params


def run_gradient_boosting(clf,fit_params,X_trn,y_trn,X_val,y_val,is_lgbm=False):
 import lightgbm as lgb
 if len(fit_params) > 0:
     _ = clf.fit(X_trn, y_trn, eval_set = [(X_val, y_val)], **fit_params)
 else:
     if is_lgbm:
         _ = clf.fit(X_trn, y_trn, eval_set = [(X_val, y_val)],callbacks=[
            lgb.early_stopping(stopping_rounds=3),
        ])
     else:
         _ = clf.fit(X_trn, y_trn, eval_set = [(X_val, y_val)])

 ### Instead of directly predicting the classes we will obtain the probability of positive class.
 preds_train = clf.predict_proba(X_trn)[:, 1]
 preds_valid = clf.predict_proba(X_val)[:, 1]
 return preds_train,preds_valid





