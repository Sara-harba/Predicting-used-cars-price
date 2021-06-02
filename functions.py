#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# A function to detect outliers
def detect_outliers(data):
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    return lower, upper


#A function to fill categorical columns with the most frequent value
def fill_cat(text):
    most_frequent = text.value_counts().index.tolist()[0]
    col = text.fillna(most_frequent)
    return col

#Choose few categories from the model which represent the hgighest count in the column
ls = ['f-150','silverado','1500','camry','accord','civic','escape','altima','explorer','equinox','wrangler','corolla',
     'mustang','malibu','fusion','tacoma','grand cherokee','focus','2500','grand caravan','tahoe','cr-v','cruze','sonata',
      'impala','elantra','jetta','sentra','prius','rav4','odyssey','outback','edge','sierra','rogue','charger','forester',
     'traverse','camaro','pilot','suburban','3500','sienna','corvette','f-250','highlander','soul','acadia','tundra',
      'cherokee','journey','optima','4runner','taurus','town & country','expedition','passat','murano','impreza','challenger',
      '300','fusion se','c-class','enclave','200','colorado','escalade','ranger','durango','pathfinder','sorento','liberty',
      'maxima','f150','patriot','santa','yukon','f250','versa','focus','tacoma','terrain','mdx','a4','cts','x5','compass',
      'yukon','lacrosse','3 series']   


#A function to create a dictionary to replace the values
def replace(text):
    dic = {}
    for value in text:
        for i in ls:
            if i in value:
                dic[value]=i
                
    return dic

#group the features that their count is less than 1000 into a group called other
def features_list(text, num):
    features_list = []
    for index, name in enumerate(text.value_counts().index.tolist()):
        values = text.value_counts()[index]
        if values < num:
            features_list.append(name)
    return features_list


# In[3]:


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler


lr = LinearRegression()
imputer = SimpleImputer(strategy="median")
transformer = RobustScaler()
scaler = StandardScaler()
cat_encoder = OneHotEncoder(handle_unknown= 'ignore')

#define the pipeline for transforming the columns 
def pipeline(X_num, Y_num, X_train):
    num_pipeline = Pipeline([('imputer', imputer), ('scaler', scaler)])
    
    num_attribs = list(X_num)
    cat_attribs = list(Y_num)
    
    pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_encoder, cat_attribs)])
    
    train_prepared = pipeline.fit_transform(X_train)
    return train_prepared    

def display_scores(scores):
    print("Scores:", np.round(scores))
    print("Mean: ", np.round(scores.mean()))
    print("Stdev:", np.round(scores.std()))
    

#bin the years in the year column into three bins 
def bin_years(column):
    new_column = pd.cut(column, bins=[1900,2000,2010,2021], labels=["Old", "Fair", "New"])
    return new_column

