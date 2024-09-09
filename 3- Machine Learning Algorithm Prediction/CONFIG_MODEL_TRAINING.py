#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is a config file for the models and datasets being tested

"""
import os
import pandas as pd
import numpy as np

#Relative paths to datasets
RFE_RFR =  os.path.join(os.path.dirname(__file__), '../', "2- Feature Selection Algorithms/FEATURE_SELECTION_DATASETS/RFE_RFR.csv")
RFE_GBR = os.path.join(os.path.dirname(__file__), '../..', "2- Feature Selection Algorithms/FEATURE_SELECTION_DATASETS/RFE_GBR.csv")
RFB = os.path.join(os.path.dirname(__file__), '../', "2- Feature Selection Algorithms/FEATURE_SELECTION_DATASETS/RFB.csv")
RandomForestEnsemble = os.path.join(os.path.dirname(__file__), '..', "2- Feature Selection Algorithms/FEATURE_SELECTION_DATASETS/EnsembleFinal.csv")

#Meta data for banks
bank_names = pd.read_excel(os.path.join(os.path.dirname(__file__), '../', "1- FDIC Data Gathering/bank_names.xlsx"), sheet_name="Sheet1")
failed_banks = pd.read_csv(os.path.join(os.path.dirname(__file__), '../', "1- FDIC Data Gathering/failed_banks.csv"))
close_date = failed_banks.columns[5]
failed_banks[failed_banks.columns[5]] = pd.to_datetime(failed_banks[failed_banks.columns[5]])


TARGETFILE = RandomForestEnsemble
PROJECT_NAME = "Ensemble dataset using similar features selected by RFE-RFR and RFB"
df = pd.read_csv(TARGETFILE)

liquidated_banks = pd.read_excel(os.path.join(os.path.dirname(__file__), '../', "1- FDIC Data Gathering/Liquidated_Banks.xlsx"), sheet_name="LiquidatedBankList")
liquidated_banks['Liquidation_date'] = pd.to_datetime(liquidated_banks['Liquidation_date'])

df['REPDTE'] = pd.to_datetime(df['REPDTE_VAL'])
df[failed_banks.columns[5]] = pd.to_datetime(df[failed_banks.columns[5]])
df = df.merge(bank_names, on=['CERT'], how='left', suffixes=("","_remove"))
df = df.merge(liquidated_banks, on=['CERT'], how='left', suffixes=("","_remove"))
df = df[df['BKCLASS'].isin(['NM',"SM","N"])] #, "SB", "SI","SL","N"])] 
df.drop([i for i in df.columns if 'remove' in i],
    axis=1, inplace=True)
df.drop_duplicates(inplace=True)

failed_banks = pd.read_csv(os.path.join(os.path.dirname(__file__), '../', "1- FDIC Data Gathering/failed_banks.csv"))
failed_banks[failed_banks.columns[5]] = pd.to_datetime(failed_banks[failed_banks.columns[5]])

#Remove banks that have been liquidated already
mask = (df["Liquidation_date"].isna()) | (df[failed_banks.columns[5]] < "2050-01-01")
df = df[mask]

#USED TO ADJUST REGRESSION TARGETS to CLASSIFICATION TARGETS
#df.loc[df['Y'] > 0, 'Y'] = 1
#LIST OF COLUMNS TO OMIT WHEN FEEDING TO MACHINE LEARNING ALGOS
FEATURE_OMISSION= ['Liquidation_date', 'CHANGECODE_DESC_LONG', 'ID','CERT', 'REPDTE','REPDTE_VAL', "BKCLASS","Y","NAME","SPECGRPN", "SUPRV_FD", df.columns.tolist()[0], failed_banks.columns[5]]
OPTIONAL_FEATURE_OMISSION = []
# Update the 'Inflation Rate' column for the filtered rows
selected_features = df[[i for i in df.columns if i not in FEATURE_OMISSION]].columns.tolist()
df[selected_features].dropna(inplace=True)
df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
df.drop_duplicates(inplace=True)



def is_within_12_months_regression(row):
    #row[close_date] = pd.to_datetime(row[close_date])
    delta = row[close_date]-row['REPDTE']
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
    
    


import pandas as pd
failed_banks = pd.read_csv(r"failed_banks.csv")

close_date = failed_banks.columns[5]


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def adjust_text_position(ax, x, y, text, delta=0.01):
    """
    Adjust text position to avoid overlap.
    :param ax: The axis object to plot on.
    :param x: X-coordinate of the point.
    :param y: Y-coordinate of the point.
    :param text: The text label.
    :param delta: Initial adjustment step.
    :return: Adjusted x, y coordinates.
    """
    while True:
        overlap = False
        for text_obj in ax.texts:
            bbox = text_obj.get_window_extent(renderer = plt.gcf().canvas.get_renderer())
            if bbox.contains(x, y):
                overlap = True
                y += delta
                break
        if not overlap:
            return x, y

def show_std_dev_plot(df_standard, repdtes):
    fig, axs = plt.subplots(nrows=3, figsize=(10, 15))
    fig.suptitle(f'Bank Failure Prediction Standard Deviation By Mean Predictions {repdtes[0]} - {repdtes[-1]} \n {PROJECT_NAME}', fontsize=12, fontweight='bold')
    
    for r, repdte in enumerate(repdtes):
        df = df_standard.loc[df_standard["REPDTE"] == repdte]
        df_filtered = df[df["Y"] > 0].copy()
        df = df[df["Y"] == 0].sample(100)

        axs[r].scatter(df["prediction_std_fail"], df["prediction_mean_fail"], color='blue')
        axs[r].scatter(df_filtered["prediction_std_fail"], df_filtered["prediction_mean_fail"], color='red')

        for i, row in df_filtered.iterrows():
            x, y = adjust_text_position(axs[r], row["prediction_std_fail"], row["prediction_mean_fail"], row["NAME"])
            axs[r].text(x, y, row["NAME"], color='red', fontsize=10)
    
        axs[r].set_xlabel("Prediction Std Fail")
        axs[r].set_ylabel("Prediction Mean Fail")
        axs[r].grid(True)
    
    plt.tight_layout(pad=4.0)
    
    plt.show()

# def show_std_dev_plot(df_standard,repdtes):
    
#     fig, axs = plt.subplots(nrows=3, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
#     fig.suptitle(f'Bank Failure Prediction Standard Deviation By Mean Predictions {repdtes[0]} - {repdtes[-1]}', fontsize=12, fontweight='bold')
    
#     for r, repdte in enumerate(repdtes):
#         df = df_standard.loc[(df_standard["REPDTE"] == repdte)]
#     # Filter the DataFrame for rows where Y > 0
#         df_filtered = df[df["Y"] > 0].copy()
#         df = df[df["Y"]==0].sample(100)
#     # Plotting
#         #axs[r].figure(figsize=(8, 6))
#         axs[r].scatter(df["prediction_std_fail"], df["prediction_mean_fail"], color='blue')
#         axs[r].scatter(df_filtered["prediction_std_fail"], df_filtered["prediction_mean_fail"], color='red')
#         #axs[r].legend()
#         axs[r].set_title(f"{repdte}")
#     # Labeling points
#         for i, row in df_filtered.iterrows():
#             axs[r].text(row["prediction_std_fail"], row["prediction_mean_fail"], row["NAME"], color='black', fontsize=10)
    
#         axs[r].set_xlabel("Prediction Std Fail")
#         axs[r].set_ylabel("Prediction Mean Fail")
#         #axs[r].set_title(f"Std. Dev by Mean Estimate - {repdte}")
#         axs[r].grid(True)
#     plt.tight_layout(pad=4.0)
#     plt.show()

#Average MAE
# RF Recursive Elimination: .557
# Gradient Boosting Regression Elim:
    




def is_within_6_months(row):
    delta = row['REPDTE_VAL'] - row[close_date]
    months_difference = delta.days / 30  # Assuming 30 days per month
    return abs(months_difference) <= 6
#import pandas as pd
#failed_banks = pd.read_csv(r"/mnt/c/Users/Wyatt Schwanbeck/source/repos/ExtractFDICBankData/ExtractFDICBankData/bin/Debug/failed_banks.csv")
#failed_banks[failed_banks.columns[5]] = pd.to_datetime(failed_banks[failed_banks.columns[5]])
# Define the directory where your CSV files are located
# directory = r'/mnt/c/Users/Wyatt Schwanbeck/source/repos/ExtractFDICBankData/ExtractFDICBankData/bin/Debug'
# #df.drop(df.columns.tolist()[0], inplace=True)
# #lIST OF COLUMNS TO OMIT WHEN FEEDING TO RANDOM FOREST BORUTA

# FEATURE_OMISSION= ['ID','REPDTE_VAL', "BKCLASS","Y"]



# RFB_COLS = pd.read_csv(RandomForestBorutaDataSet).columns.tolist()
# TBF_COLS = pd.read_csv(TreeBasedFeatureSelection).columns.tolist()#.drop(FEATURE_OMISSION).columns.tolist()
# RF_COLS = pd.read_csv(RecursiveFeatureElimination).columns.tolist()#.drop(FEATURE_OMISSION).columns.tolist()



# print("Random Forest Boruta Column Count: {} ".format(len(RFB_COLS)) )
# print("Tree Based Feature Selection Column Count: {} ".format(len(TBF_COLS)) )
# print("Recursive Feature Removal Column Count: {} ".format(len(RF_COLS)) )


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


def merge_on_common_columns(file1, file2,file3):
    # Read the dataframes
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Find common columns
    common_columns = set(df1.columns) & set(df2.columns) & set(df3.columns)

    # Subset the dataframes to only include common columns
    df1_common = df1[list(common_columns)]
    df2_common = df2[list(common_columns)]
    df3_common = df3[list(common_columns)]

    # Merge the dataframes horizontally
    merged_df = pd.concat([df1_common, df2_common, df3_common], axis=1)

    # Optional: if you want to remove duplicate columns after merge
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    merged_df.to_csv(RandomForestEnsemble)
    
    return merged_df


def calculate_average_sampled_precision_recall(df_standard, test_date_start,test_date_end, sample_count = 25):
    #Sample 100 non-failing banks alongside actual test failing banks 100 times. Calculate average precision, recal, and fmeasure
    #train_df = df_standard[df_standard['REPDTE'] < test_date_start][df_standard['REPDTE'] <= test_date_end]
    test_df = df_standard[df_standard['REPDTE'] >= test_date_start][df_standard['REPDTE'] <= test_date_end]
    total_precision = 0
    total_recall = 0
    total_fmeasure = 0
    precision, recall, fmeasure = calculate_precision_recall_fMeasure(test_df)
    actual_failing_banks = test_df[test_df["Y"]>0]
    
    for i in range(sample_count):
        actual_non_failing_banks = test_df[test_df["Y"]==0].sample(len(actual_failing_banks)*10)
        sample_df = pd.concat([actual_failing_banks, actual_non_failing_banks],ignore_index=True)
        precision, recall, fmeasure = calculate_precision_recall_fMeasure(sample_df)
        total_precision+=precision
        total_recall += recall
        total_fmeasure += fmeasure
    return total_precision/sample_count, total_recall/sample_count, total_fmeasure/sample_count
    
    
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

    # Optional: if you want to remove duplicate columns after merge
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    merged_df.to_csv(RandomForestEnsemble)
    
    return merged_df

#merge_on_common_columns(RFB, RFE_RFR)
#merge_on_common_columns(RF_Recursive,RandomForestBorutaDataSet)
def calculate_precision_recall_fMeasure(df, threshold=.50):
    # True Positives (TP): Correctly predicted positives
    TP = len(df[(df["prediction_mean_fail"] > threshold) & (df["Y"] > threshold)])
    
    # False Positives (FP): Incorrectly predicted as positives
    FP = len(df[(df["prediction_mean_fail"] > threshold) & (df["Y"] <= threshold)])
    
    # False Negatives (FN): Incorrectly predicted as negatives
    FN = len(df[(df["prediction_mean_fail"] <= threshold) & (df["Y"] > threshold)])

    if(TP+FP!=0):
    ## Precision Calculation
        precision = TP / (TP + FP)
    else:
        precision = 0
    if(TP+FN!=0):
        # Recall Calculation
        recall = TP / (TP + FN)
    else:
        recall = 0
    # F-Measure Calculation
    if(precision+recall!=0):
        fmeasure = (2 * precision * recall) / (precision + recall)
    else:
        fmeasure = 0
    return precision, recall, fmeasure