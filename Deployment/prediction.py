import pickle
# import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
import sklearn
import numpy as np

# loading the trained model
pickle_in = open('catboost_model.pkl', 'rb') 
classifier = pickle.load(pickle_in)

def cont_to_cat(df,cat_col,cont_col):
  temp = []
  no_rows = df.shape[0]
  for i in range(0, no_rows):
    if int(df[cont_col][i]) > 1:
      temp.append(1)
    else:
      temp.append(0)
  
  df[cat_col] = pd.Series(temp).astype('object')
  return df

# For getting data types of dataframes
def get_dtypes(df):
  columns = df.columns
  num_cols = []
  cat_cols = []
  for column in columns:
    if (is_numeric_dtype(df[column])):
      num_cols.append(column) 
    else:
      cat_cols.append(column)

  return num_cols,cat_cols

def one_hot_encoding(data,cat_feats):
  one_hot = pd.get_dummies(data[cat_feats])
  one_hot = one_hot.astype('object')
  data.drop(cat_feats,axis=1,inplace=True)
  data = pd.concat([data,one_hot],axis=1)
  return data

def transform_data(test):
  columns = ['gender','region','department','recruitment_channel']

  for column in columns:
    d = {}
    for value in test[column].unique():
      d[value] = test.loc[test[column] == value][column].shape[0]

    test[column+'_count'] = test[column].apply(lambda x:d[x])

  bins = [-1,2,5,7,10,100]
  labels = ['0-2','3-5','5-7','8-10','>10']
  test['length_of_service_binned'] = pd.cut(test['length_of_service'], bins,labels=labels)

  bins = [-1, 29, 39, 49,100]
  labels = ['<30','30-39','40-49','>=50']
  test['age_binned'] = pd.cut(test['age'], bins,labels=labels)

  test['KPIs_met >80%'] = test['KPIs_met >80%'].apply(lambda x:str(x))
  test['awards_won?'] = test['awards_won?'].apply(lambda x:str(x))

  test = cont_to_cat(test,'is_multiple_training_completed','no_of_trainings')


  high_freq = ['region_2','region_22','region_7','region_15','region_13','region_26']
  temp = []
  no_rows = test.shape[0]
  for i in range(0, no_rows):
    if test['region'][i] in high_freq:
      temp.append(1)
    else:
      temp.append(0)

  test['region_high_employees'] = pd.Series(temp).astype('object')

  test.drop(['region'],axis=1,inplace=True)
  num_cols, cat_cols = get_dtypes(test)

  nominal = ['department','recruitment_channel','length_of_service_binned','age_binned','gender'] 
  ordinal = ['education', 'KPIs_met >80%', 'awards_won?', 'is_multiple_training_completed', 'region_high_employees']

  test = one_hot_encoding(test,nominal)

  test['education'] = test['education'].map({'null':0,'Below Secondary':1,"Bachelor's":2,"Master's & above":3})


  for variable in test.columns:
    if(test[variable].dtypes == "object"): 
      test[variable] = test[variable].apply(lambda x:int(x))

  #Dropping gender_f variable.
  if 'gender_f' in test.columns:
    test.drop(['gender_f'],axis=1,inplace=True)

  test['previous_year_rating'] = test['previous_year_rating'].apply(lambda x:int(x))

  test['log_avg_training_score'] = pd.DataFrame(np.log(test['avg_training_score'] * 5+ 100))
  test['log_no_of_trainings'] = pd.DataFrame(np.log(test['no_of_trainings'] * 5 + 100))

  to_add = ['log_no_of_trainings','log_avg_training_score']
  to_drop = ['no_of_trainings','avg_training_score']
  test.drop(to_drop,axis=1,inplace=True)

  # changing num_cols values
  for col in to_drop:
    num_cols.remove(col)

  for col in to_add:
    num_cols.append(col)

  test[num_cols] = (test[num_cols] - test[num_cols].min())/(test[num_cols].max()-test[num_cols].min())
  return test

def predict(Gender,Age,Department,Region,Education,Recruitment_channel,Num_training_completed,Previous_year_rating,Length_of_service,KPIs_met,Awards_won,Avg_training_score):
    if KPIs_met == "No":
        KPIs_met = '0'
    else:
        KPIs_met = '1'

    if Awards_won == "No":
        Awards_won = '0'
    else:
        Awards_won = '1'

    data = {
    'department':[Department],
    'region':[Region],
    'education':[Education],
    'gender':[Gender],
    'recruitment_channel':[Recruitment_channel],
    'no_of_trainings':[Num_training_completed],
    'age':[Age],
    'previous_year_rating':[Previous_year_rating],
    'length_of_service':[Length_of_service],
    'KPIs_met >80%':[KPIs_met],
    'awards_won?':[Awards_won],
    'avg_training_score':[Avg_training_score]
    }

    test = transform_data(pd.DataFrame.from_dict(data))

    columns = ['education', 'age', 'previous_year_rating', 'length_of_service','KPIs_met >80%', 'awards_won?', 'gender_count', 'region_count',
'department_count', 'recruitment_channel_count','is_multiple_training_completed', 'region_high_employees','department_Analytics', 'department_Finance', 'department_HR','department_Legal', 'department_Operations', 'department_Procurement','department_R&D', 'department_Sales & Marketing','department_Technology', 'recruitment_channel_other','recruitment_channel_referred', 'recruitment_channel_sourcing','length_of_service_binned_0-2', 'length_of_service_binned_3-5','length_of_service_binned_5-7', 'length_of_service_binned_8-10','length_of_service_binned_>10', 'age_binned_<30', 'age_binned_30-39','age_binned_40-49', 'age_binned_>=50', 'gender_m','log_avg_training_score', 'log_no_of_trainings']
    
    # modify test df, sorting columns in the order used by our model
    test_mod = pd.DataFrame(test,columns = columns)

    #0.25 is the optimal threshold which has been calculated while modelling
    probability_promotion = (classifier.predict_proba(test_mod)[:, 1] > 0.25) * 1
    if probability_promotion >= 0.25:
        pred = "Promoted"
    else:
        pred = "Not Promoted"

    return pred

