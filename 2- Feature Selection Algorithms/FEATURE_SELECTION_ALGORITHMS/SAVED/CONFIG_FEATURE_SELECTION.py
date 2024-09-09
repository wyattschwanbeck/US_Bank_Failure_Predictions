#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is a config file for the feature compilation datasets

"""
import os
import pandas as pd

RFE_RFR =  os.path.join(os.path.dirname(__file__), '..', "FEATURE_SELECTION_DATASETS/RFE_RFR.csv")
RFE_GBR = os.path.join(os.path.dirname(__file__), '..', "FEATURE_SELECTION_DATASETS/RFE_GBR.csv")
RFB = os.path.join(os.path.dirname(__file__), '..', "FEATURE_SELECTION_DATASETS/RFB.csv")

RandomForestEnsemble = os.path.join(os.path.dirname(__file__), '..', "FEATURE_SELECTION_DATASETS/EnsembleFinal.csv")

All_Files_Directory = os.path.join(os.path.dirname(__file__), '../..', "1- FDIC Data Gathering/All_Data")

failed_banks = pd.read_csv(os.path.join(os.path.dirname(__file__), '../..', "1- FDIC Data Gathering/failed_banks.csv"))
close_date = failed_banks.columns[5]
failed_banks[failed_banks.columns[5]] = pd.to_datetime(failed_banks[failed_banks.columns[5]])

#LIST OF COLUMNS TO OMIT WHEN FEEDING TO FEATURE SELECTION MODELS
FEATURE_OMISSION= ['ID','CERT', 'REPDTE','REPDTE_VAL',"Y","BKCLASS", failed_banks.columns[5]]

OPTIONAL_FEATURE_OMISSION = []

ratio_features = pd.read_csv(os.path.join(os.path.dirname(__file__), '../..', "1- FDIC Data Gathering/extracted_properties_with_formulas.csv"))

liquidated_banks = pd.read_excel(os.path.join(os.path.dirname(__file__), '../..', "1- FDIC Data Gathering/Liquidated_Banks.xlsx"), sheet_name="LiquidatedBankList")
liquidated_banks['Liquidation_date'] = pd.to_datetime(liquidated_banks['Liquidation_date'])

def is_within_12_months_regression(row):
    delta = row[close_date]-row['REPDTE_VAL']
    months_difference = delta.days / 30  # Assuming 30 days per month
    if(months_difference>12):
        return 0
    else:    
        return delta.days/(365)

   
def is_within_6_months_regression(row):
    delta = row[close_date]-row['REPDTE_VAL']
    months_difference = delta.days / 30  # Assuming 30 days per month
    if(months_difference>6):
        return 0
    else:    
        return delta.days/(30*6)
    
def is_within_6_months(row):
    delta = row['REPDTE_VAL'] - row[close_date]
    months_difference = delta.days / 30  # Assuming 30 days per month
    return abs(months_difference) <= 6

def find_common_columns(RFB_file, TBF_file, RF_file):
    # Read the datasets and get the column names
    RFB_COLS = pd.read_csv(RFB_file).columns.tolist()
    TBF_COLS = pd.read_csv(TBF_file).columns.tolist()
    RF_COLS = pd.read_csv(RF_file).columns.tolist()

    # Convert lists to sets for efficient comparison
    RFB_set = set(RFB_COLS)
    TBF_set = set(TBF_COLS)
    RF_set = set(RF_COLS)

    # Find intersection (common elements) among the sets
    common_columns = RFB_set.intersection(TBF_set, RF_set)

    # Convert the set back to a list if needed
    return list(common_columns)


def merge_on_common_columns(file1, file2):
    # Read the dataframes
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    #df3 = pd.read_csv(file3)

    # Find common columns
    common_columns = set(df1.columns) & set(df2.columns) # & set(df3.columns)

    # Subset the dataframes to only include common columns
    df1_common = df1[list(common_columns)]
    df2_common = df2[list(common_columns)]
    #df3_common = df3[list(common_columns)]

    # Merge the dataframes horizontally
    merged_df = pd.concat([df1_common, df2_common], axis=1)

    #Remove duped columns
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    merged_df.to_csv(RandomForestEnsemble)
    
    return merged_df

