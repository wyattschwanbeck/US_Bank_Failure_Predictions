# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 20:11:56 2023
boruta.fit(balanced_set.drop(FEATURE_OMISSION, axis=1).values, balanced_set["Y"].values)
        
# Define a function to check if REPDTE is within 6 months of Closing Date
# Get selected feature names
        cols = balanced_set.drop(FEATURE_OMISSION, axis=1, inplace = False).columns
        selected_indices = boruta.support_
        selected_features = np.array(cols)[np.where(boruta.ranking_== 1)]
        
        print(selected_features)
        selected_features = selected_features.tolist()
@author: Wyatt Schwanbeck
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np
from boruta import BorutaPy

from CONFIG_FEATURE_SELECTION import is_within_6_months, FEATURE_OMISSION, failed_banks, All_Files_Directory,close_date,ratio_features, OPTIONAL_FEATURE_OMISSION, RFB

min_features_to_select = 1  # Minimum number of features to consider

total_features = FEATURE_OMISSION.copy()
combined_df = pd.DataFrame()
for i in range(1, 24):  # Assuming you have files named from 1 to 23
    print("Reading {0} File".format(i))
    filename = f'TotalResults{i}.csv'
    file_path = os.path.join(All_Files_Directory, filename)
    columns = []
    if os.path.exists(All_Files_Directory):

        df = pd.read_csv(file_path)#,usecols=columns)
        filtered_columns = [col for col in df.columns if col in ratio_features["Property Name"].tolist()] #df.filter(regex='R$', axis=1).columns.tolist()
        if(len(filtered_columns)>0):
            filtered_columns.extend(FEATURE_OMISSION)
            filtered_columns = list(set(filtered_columns))
            df = df[df['BKCLASS'].isin(['NM',"SM", "SB", "SI","SL","N"])]
            
            df = df.merge(failed_banks[['CERT',failed_banks.columns[5]]], on='CERT', how='left')
            df['REPDTE_VAL'] = pd.to_datetime(df['REPDTE'], format='%Y%m%d')

            df['Y'] = df.apply(is_within_6_months, axis=1)
            
            
            df[close_date].fillna("2050-01-01", inplace=True)
    
            df.replace([np.nan], 0, inplace=True)
            df = df[filtered_columns]

            combined_df = df
            
            train = combined_df[combined_df['REPDTE_VAL'] < '2024-01-01'][combined_df['REPDTE_VAL'] > '2000-01-01' ].copy()
            
            rf = RandomForestClassifier(n_jobs=-1)#, class_weight="balanced")#, random_state=42)
   
           
            # Define Boruta feature selection
            boruta = BorutaPy(rf, n_estimators='auto',verbose=2)#, random_state=42)
            
            balanced_set = train[train["Y"]>0.0].copy()
            balanced_set = pd.concat([train[train["Y"]==0].sample(len(balanced_set)), balanced_set],ignore_index=True)
            feature_drop = balanced_set.filter(OPTIONAL_FEATURE_OMISSION)
            balanced_set.drop(feature_drop, axis=1, inplace = True)
            boruta.fit(balanced_set.drop(FEATURE_OMISSION, axis=1).values, balanced_set["Y"].values)
                    
            # Define a function to check if REPDTE is within 6 months of Closing Date
            # Get selected feature names
            cols = balanced_set.drop(FEATURE_OMISSION, axis=1, inplace = False).columns
            selected_indices = boruta.support_
            selected_features = np.array(cols)[np.where(boruta.ranking_== 1)]
            
            print(selected_features)
            selected_features = selected_features.tolist()
    
            print(selected_features)
            selected_features = selected_features.tolist()
    
            total_features = FEATURE_OMISSION.copy()
            
            total_features.extend(selected_features.copy())
            total_features= list(set(total_features))


for i in range(1, 24):  # Assuming you have files named from 1 to 23
    print("Reading {0} File".format(i))
    filename = f'TotalResults{i}.csv'
    file_path = os.path.join(All_Files_Directory, filename)
    columns = []
    if os.path.exists(file_path):

        df = pd.read_csv(file_path)
        df = df.merge(failed_banks[['CERT',failed_banks.columns[5]]], on='CERT', how='left')
        df['REPDTE_VAL'] = pd.to_datetime(df['REPDTE'], format='%Y%m%d')
        close_date = failed_banks.columns[5]
        
        df['Y'] = df.apply(is_within_6_months, axis=1)
        df.drop_duplicates(inplace=True)
            
        df[close_date].fillna("2050-01-01", inplace=True)

        matching_total_features = df[[i for i in total_features if i in df.columns]].columns.tolist()
        
        if i > 1:
            
            combined_df = combined_df.merge(df[matching_total_features], on=['CERT','REPDTE'], how='inner',suffixes=('', '_remove'))
            # Drop duplicated columns
            combined_df.drop([i for i in combined_df.columns if 'remove' in i],
               axis=1, inplace=True)
            
           
        else:
            combined_df = df[matching_total_features]

combined_df.to_csv(RFB)