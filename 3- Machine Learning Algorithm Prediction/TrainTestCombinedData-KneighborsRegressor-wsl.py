# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:34:26 2023

@author: Wyatt Schwanbeck
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
#import TestBayesianOptimization as TBO
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
# Load dataset
from CONFIG_MODEL_TRAINING import TARGETFILE, PROJECT_NAME, \
    calculate_precision_recall_fMeasure,show_std_dev_plot, \
        is_within_12_months_regression, \
        calculate_average_sampled_precision_recall, \
        df, FEATURE_OMISSION, failed_banks,selected_features
        
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score


df_standard = df.copy() # pd.concat(sections,ignore_index=True) #df.copy() #

import os

df_standard['REPDTE'] = pd.to_datetime(df_standard['REPDTE'])
test = df_standard[df_standard['REPDTE'] >= '2022-09-30']

#test.to_csv(r"/mnt/c/Users/Wyatt Schwanbeck/OneDrive/Documents/CSCI 794 - Masters Thesis/Bank Failure/TestDataWhole.csv")
train = df_standard[df_standard['REPDTE'] < '2022-09-30'][df_standard['REPDTE'] > '1992-01-01' ].copy()


import pandas as pd


#X_train = train.drop(FEATURE_OMISSION, axis=1)

#X_train.dropna(inplace=True)
#X_train = train[features]
#y_train = train['Y']

#X_test = test.drop(FEATURE_OMISSION,axis=1)
#y_test = test['Y']
#X_train.reset_index(drop=True, inplace=True)
#y_train.reset_index(drop=True, inplace=True)
#train.reset_index(drop=True, inplace=True)
#test.reset_index(drop=True, inplace=True)
# Set the number of folds and create a StratifiedKFold object
#rf = RandomForestRegressor(n_jobs=-1)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

import numpy as np


# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.

balanced_set_Y = train[train["Y"]>0]
balanced_test_set_Y = test[test["Y"]>0]


balanced_set = pd.concat([train[train["Y"]==0].sample(len(balanced_set_Y), random_state=0), balanced_set_Y],ignore_index=True)
balanced_sample_set = balanced_set.sample(int(len(balanced_set)*.5), random_state=42)
balanced_set.drop(balanced_sample_set.index)
#X_train, X_val, y_train, y_val = train_test_split(balanced_sample_set[selected_features], balanced_sample_set["Y"], test_size=0.2)


data, targets = balanced_set[selected_features], balanced_set["Y"]
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK


# #Create Hyperparameter space
# space= {'n_neighbors': hp.choice('n_neighbors',np.arange(1, 500, dtype=int)), #hp.choice('n_neighbors', range(50, 250, 1)),
#         #'leaf_size':hp.choice('leaf_size', np.arange(1, 100, dtype=int))
#         #'p':hp.uniform('p', 0,2),
#         #'weights':hp.choice('weights', ["uniform", 'distance']),
#         #'max_samples':hp.uniform('max_samples', 0.01,1),
#        }
rf = KNeighborsRegressor()

from sklearn.metrics import mean_squared_error
#Define Objective Function
def objective(space):
    
    rf = KNeighborsRegressor(**space)

    
    # fit Training model
    rf.fit(data, targets)
    
    # Making predictions and find RMSE
    y_pred = rf.predict(balanced_test_set_Y[selected_features])
    mse = mean_squared_error(balanced_test_set_Y["Y"],y_pred)
    rmse = np.sqrt(mse)
    
    
    # Return RMSE
    return rmse




#Surrogate Fn
trials = Trials()
# params = fmin(objective,
#     space=space,
#     algo=tpe.suggest,
#     max_evals=100,
#     trials=trials)

#print(best)
print(trials.results)
#print(TBO.Colours.yellow("--- Optimizing SVM ---"))
#optimize_svc(data, targets)

#print(TBO.Colours.green("--- Optimizing Random Forest ---"))
#rf_params = TBO.optimize_rfc(data, targets)
#rf = RandomForestRegressor(n_estimators=int(params["n_estimators"]), min_samples_leaf=params["min_samples"],max_samples= params["max_samples"])
rf = KNeighborsRegressor()#n_neighbors=int(params["n_neighbors"])) #, leaf_size=params["leaf_size"])#,weights= float(params["weights"]))
import pandas as pd


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

dataset_size = len(test)
batch_size = 128
#train_size = int(dataset_size * 0.85)


# And then fit and transform your data:
scaled_features_test = test[selected_features].copy()
scaled_features_balanced = balanced_set[selected_features].values

num_epochs = 1
#prob_bnn_model = create_probablistic_bnn_model()
#prob_final_model = create_probablistic_bnn_model(train_size)
features = balanced_set[selected_features].to_numpy()
#features.reset_index(inplace=True)
labels = balanced_set["Y"].to_numpy()

# Convert labels to one-hot encoding if necessary
#one_hot_labels = tf.keras.utils.to_categorical(labels)
num_folds = 50
skf = KFold(n_splits=num_folds)#, shuffle=True, random_state=43)
# Train the model using stratified forward chaining
val_indexes = []
#backupModel =create_probablistic_bnn_model(train_size)
#prob_bnn_model.save_weights(r"/mnt/c/Users/Wyatt Schwanbeck/source/repos/temp_training2")
prior_pred = 0
median_spread = 100
true_mask = df_standard[df_standard["Y"]>0][df_standard["REPDTE"]>="9-30-2022"]["Y"]==True
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
selected_output_features = selected_features.copy()#.tolist()

selected_output_features.extend(FEATURE_OMISSION)
#selected_output_features.append("prediction_mean_non-fail")
#selected_output_features.append("prediction_std_non-fail")
#selected_output_features.append("prediction_std_fail")
selected_output_features.append("prediction_mean_fail")
selected_output_features.append("Prediction Mean Rounded")
selected_output_features.append("Will or Won't fail")
selected_output_features= selected_output_features[0:-1].copy()

mask = df['REPDTE'] > pd.to_datetime('2020-01-01')

#prob_bnn_model.load_weights(r"/mnt/c/Users/Wyatt Schwanbeck/source/repos/temp_training_improvement-{fold}")
testDates = ["9-30-2022","12-31-2022", "3-31-2023"]
bkclass_specgrp= [#("NM","All Other Over 1 Billion",["12-31-2022", "3-31-2023"]), /
                  #("SM","All Other Over 1 Billion",["9-30-2022","12-31-2022"]), /
                 #("NM","Commercial Lending Specialization",["9-30-2022","12-31-2022"]), /
                     ("*", "*", ["9-30-2022","12-31-2022", "3-31-2023"])]
# Function to compute RMSE
def compute_rmse(predictions, actuals):
    return np.sqrt(((predictions - actuals) ** 2).mean())

df_std_temp= df_standard[selected_features].copy()
test_input = df_std_temp[selected_features]

#rmse = run_experiment(prob_bnn_model, negative_loglikelihood, train_data_fold, val_data_fold, num_epochs=num_epochs)
#rf.fit(balanced_set[selected_features],balanced_set["Y"])

# it's a good practice to use `.loc` for conditional subsetting
min_mae = 1000
for fold in range(num_folds):
    print(f"Training fold: {fold+1}/{num_folds}")
    
    # Get the training and validation data for the current fold
    
    balanced_set = pd.concat([train[train["Y"]==0].sample(len(balanced_set_Y), random_state=fold), balanced_set_Y],ignore_index=True)
    balanced_sample_set = balanced_set.sample(int(len(balanced_set)*.2), random_state=fold)
    balanced_set.drop(balanced_sample_set.index)
    features = balanced_set[selected_features]
    labels = balanced_set["Y"]
    x_train, x_val = balanced_set[selected_features], balanced_sample_set[selected_features]
    y_train, y_val = balanced_set["Y"], balanced_sample_set["Y"]
    
    rf.fit(x_train,y_train)
    
    df_standard['prediction_mean_fail'] = rf.predict(test_input)#rf.predict_proba(test_input)[0:,1]#Returns probability to classify as 0 and 1, taking just probabilities of default
    df['prediction_mean_fail'] = df_standard['prediction_mean_fail']
    #df_standard[['prediction_std_fail']] = prediction_distribution
    # Splitting dataframe into train and test
    train_df = df_standard[df_standard['REPDTE'] < '2022-09-30'].copy()
    test_df = df_standard[df_standard['REPDTE'] >= '2022-09-30'].copy()
    # Compute RMSE for train and test
    train_rmse = compute_rmse(train_df['prediction_mean_fail'], train_df['Y'])
    test_mae = mean_absolute_error(test_df[test_df['Y']>0]['prediction_mean_fail'],test_df[test_df['Y']>0]['Y'])
    test_mae_nonFailed = mean_absolute_error(test_df[test_df['Y']==0]['prediction_mean_fail'],test_df[test_df['Y']==0]['Y'])
    #print(f"Test MAE Non-Failed Banks : {test_mae_nonFailed}")
    #print(f"Train MAE: {train_mae}")
    #print(f"Test MAE: {test_mae}")
    if(test_mae+test_mae_nonFailed<min_mae):
        print(f"Test MAE Non-Failed Banks : {test_mae_nonFailed}")
        #print(f"Train MAE: {train_mae}")
        print(f"Test MAE: {test_mae}")
        min_mae = test_mae+test_mae_nonFailed

    
    print(f"Test Mean Absolute Error Failed Banks: {test_mae}")
    for combo in bkclass_specgrp:
        #fig, ax = plt.subplots(nrows=3)
        if(combo[0] == "*"):
            #TrueFailPred = df_standard.loc[(df_standard["Y"] == True) & (df_standard["REPDTE"] == testDate), "prediction_mean_fail"]
            #NonFailPred = df_standard.loc[(df_standard["Y"] == False) & (df_standard["REPDTE"] == testDate), "prediction_mean_fail"]
            fig, axs = plt.subplots(nrows=3, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
            fig.suptitle(f'Bank Failure Predictions \n {PROJECT_NAME} - Default K Neighbors Regressor - Fold {fold}', fontsize=10, fontweight='bold')
        else:
            
            #TrueFailPred = df_standard.loc[(df_standard["Y"] == True) & (df_standard["REPDTE"] == testDate) & (df_standard['BKCLASS'] == combo[0]) & (df_standard['SPECGRPN'] == combo[1]), "prediction_mean_fail"]
            #NonFailPred = df_standard.loc[(df_standard["Y"] == False) & (df_standard["REPDTE"] == testDate) & (df_standard['BKCLASS'] == combo[0]) & (df_standard['SPECGRPN'] == combo[1]), "prediction_mean_fail"]
            fig, axs = plt.subplots(nrows=2, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
            fig.suptitle(f'Bank Failure Predictions - {combo[0]} - {combo[1]}', fontsize=12, fontweight='bold')
        
        for t, testDate in enumerate(combo[2]):
            if(combo[0] == "*"):
                TrueFailPred = df_standard.loc[(df_standard["Y"] >0) & (df_standard["REPDTE"] == testDate), "prediction_mean_fail"]
                NonFailPred = df_standard.loc[(df_standard["Y"] ==0) & (df_standard["REPDTE"] == testDate), "prediction_mean_fail"]
                #LiquidatedPred = df_standard.loc[(df_standard["REPDTE"] == testDate) & (df_standard["CHANGECODE_DESC_LONG"]=="Closed voluntarily and liquidated assets."), "prediction_mean_fail"]
                #fig, axs = plt.subplots(nrows=3, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
                #fig.suptitle(f'Bank Failure Predictions - All State Regional Banks FDIC members and non-members - k-fold {fold}', fontsize=12, fontweight='bold')
            else:
                
                TrueFailPred = df_standard.loc[(df_standard["Y"] > 0) & (df_standard["REPDTE"] == testDate) & (df_standard['BKCLASS'] == combo[0]) & (df_standard['SPECGRPN'] == combo[1]), "prediction_mean_fail"] #(df_standard["CHANGECODE_DESC_LONG"]==0)
                NonFailPred = df_standard.loc[(df_standard["Y"] == 0) & (df_standard["REPDTE"] == testDate) & (df_standard['BKCLASS'] == combo[0]) & (df_standard['SPECGRPN'] == combo[1]), "prediction_mean_fail"]
                #fig, axs = plt.subplots(nrows=2, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
                #fig.suptitle(f'Bank Failure Predictions - {combo[0]} - {combo[1]} - k-fold {fold}', fontsize=12, fontweight='bold')
            #LiquidatedPred = df_standard.loc[(df_standard["REPDTE"] == testDate) & (df_standard["CHANGECODE_DESC_LONG"].str.startswith("Closed")), "prediction_mean_fail"]
            colors = ['#E69F00', '#56B4E9']#, '#FF0000']
            names = [f"Non-Fail Bank Predictions", f"Actual Failed Bank Predictions"]#, "Closed Voluntarily"]  
            
            # Note that `axs[t]` is used to specify the subplot where the hist should be drawn
            n, bins, patches = axs[t].hist([NonFailPred, TrueFailPred], label=names, color=colors, log=True, bins=40)
            axs[t].legend()
            axs[t].set_title(f"Count of Prediction Rates - {testDate}")  # set title for each subplot
            # Annotating the counts above the bars
            bin_num = 0
            for count, bin_start, patch in zip(n[0], bins, patches[0]):
                if count > 0:
                    axs[t].annotate(f"{int(count)}", (bin_start, count), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
                if(bin_num==39):
                    top_false_positive_idx = NonFailPred.idxmax()
                    top_false_positive_value = NonFailPred.max()
                    top_false_positive_name = df_standard.loc[top_false_positive_idx, "NAME"]
                    # Annotate the top false positive
                    # axs[t].annotate(f"{top_false_positive_name}",
                    #    (bin_start, count), textcoords="offset points", xytext=(5,25), ha='center', fontsize=7, color="red")
                bin_num+=1
            # Also annotating for the second dataset in the histogram
            # for count, bin_start, patch in zip(n[1], bins, patches[1]):
            #     if count > 0:
            #         axs[t].annotate(f"{int(count)}", (bin_start, count), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
            for e, zipped in enumerate(zip(n[1], bins, patches[1]),1):
                count, bin_start, patch = zipped
                if count > 0:
                    axs[t].annotate(f"{int(count)}", (bin_start, count), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
                    
                    # Get the names of the institutions for the TruePred values in the current bin 
                    if(combo[0] == "*"):
                        
                        maskAnnotate = (df_standard["Y"] > 0) & (df_standard["REPDTE"] == testDate) & (df_standard["prediction_mean_fail"] >= bin_start) & (df_standard["prediction_mean_fail"] < bin_start + (bins[1] - bins[0]))
                    else:
                        
                        maskAnnotate = (df_standard["Y"] >0) & (df_standard["REPDTE"] == testDate) & (df_standard["BKCLASS"]==combo[0]) & (df_standard['SPECGRPN'] == combo[1]) & (df_standard["prediction_mean_fail"] >= bin_start) & (df_standard["prediction_mean_fail"] < bin_start + (bins[1] - bins[0]))
                    institutions = df_standard.loc[maskAnnotate, "NAME"]
                    
                    # Annotate the names, carefully adjusting positions to avoid clutter
                    # for i, institution in enumerate(institutions):
                    #     axs[t].annotate(institution, (bin_start, count * (1 + 0.1 * (i+1))), textcoords="offset points", xytext=(5,5), ha='center', fontsize=6, color='red')
                    # Annotate the names, carefully adjusting positions to avoid clutter
                    for i, institution in enumerate(institutions):
                        axs[t].annotate(institution, 
                                        (bin_start, count * (1 + 0.2 * (i+1))), 
                                        textcoords="offset points", 
                                        xytext=(5, (e%5)*25 + (i*25)), fontweight='bold',#xytext=(5 *(i+1), 10 + (10*bin_start) * (i+10+bin_start)), fontweight='bold',
                                        ha='center', 
                                        fontsize=7, 
                                        color='black')
            
            axs[t].xaxis.set_major_formatter(PercentFormatter(1))
                    
                    # Step 1: Concatenate both predictions
            all_predictions = pd.concat([TrueFailPred, NonFailPred])
            
            # Step 2: Sort in descending order
            sorted_predictions = all_predictions.sort_values(ascending=False)
            
            # Step 3: Calculate the value corresponding to the top 10% of predictions
            top_10_percent_value = sorted_predictions.iloc[int(0.2 * len(sorted_predictions))]
            
            # Note that `axs[t]` is used to specify the subplot where the hist should be drawn
            n, bins, patches = axs[t].hist([NonFailPred, TrueFailPred], label=names, color=colors, log=True, bins=40)
            # ...
            
            # Step 4: Identify which histogram bin this value lies within
            bin_index = np.digitize(top_10_percent_value, bins) - 1  # Subtract 1 to get the 0-based bin index
            
            # Step 5: Highlight or annotate this bin on the histogram
            axs[t].patches[bin_index].set_facecolor('red')  # Set bin color to red for highlighting
            #axs[t].annotate("Top 20% Predictions", (bins[bin_index], n[0][bin_index]), textcoords="offset points", xytext=(0, -15), ha='center', va='top', fontsize=6, color="red")
            
            # Draw the vertical line
            #axs[t].axvline(x=top_10_percent_value, color='red', linestyle='--')
    
        plt.tight_layout(pad=4.0)
        plt.show()

df["Prediction Mean Rounded"] = round(df["prediction_mean_fail"],2)
df["Will or Won't fail"] = df[failed_banks.columns[5]]<"2049-12-31"
df[selected_output_features][mask].to_csv(rf"/mnt/c/Users/Wyatt Schwanbeck/source/repos/Test_2022-2023-RandomForest.csv")
#time.sleep(5)
# Add a title to the entire plot
TrueFailPred = df_standard.loc[(df_standard["Y"] == True) & (df_standard["REPDTE"] >= "9-30-2022"), "prediction_mean_fail"]
NonFailPred = df_standard.loc[(df_standard["Y"] == False) & (df_standard["REPDTE"] >= "9-30-2022"), "prediction_mean_fail"]
if(NonFailPred.median()-TrueFailPred.median()<median_spread):
      
#     #prior_pred= TruePred
      #if(fold>1):
          #prob_bnn_model.save_weights(rf"/mnt/c/Users/Wyatt Schwanbeck/source/repos/temp_training-172x172x172-{fold}")
      median_spread = NonFailPred.median()-TrueFailPred.median()
      #print(f"*********************Prediction improved NonFail Median: {NonFailPred} True Pred Median {TrueFailPred}*****************************")
      
      
    #else:
    #     time.sleep(1)
         
    #print(f"*********************Prediction NonFail Median: {NonFailPred} True Pred {TrueFailPred}*****************************")
    
    #     try:
    #         prob_bnn_model.load_weights(r"/mnt/c/Users/Wyatt Schwanbeck/source/repos/temp_training_improvement")
    #     except:
    #         print("checksum error, continue")
    # NonFailPred = df_standard["prediction_mean_fail"]<=.01
    # print(f"Proportion of Non Fail Predictions: {len(TruePred==True)}")
    # if(len(TruePred==True)>=4 and len(NonFailPred==True)> len(NonFailPred==False)):
    #     break
    # input_timer = input("continue [enter]")
    # if input_timer !="":
    #     break
#prob_bnn_model.save_weights(r"/mnt/c/Users/Wyatt Schwanbeck/source/repos/temp_training_improvement-latest")
prediction_distribution = rf.predict_proba(df_standard[selected_features])[0:,1]
weightoutput = r"/mnt/c/Users/Wyatt Schwanbeck/source/repos/temp_training_improvement"
df_standard['prediction_mean_fail'] = prediction_distribution
#df_standard[['prediction_std_fail']] = prediction_distribution
TruePred = df_standard[df_standard["Y"]>0][df_standard["REPDTE"]=="12-31-2022"]["prediction_mean_fail"]
FailPred = df_standard[df_standard["Y"]==0][df_standard["REPDTE"]=="12-31-2022"]["prediction_mean_fail"].median()
#prob_bnn_model.save_weights(weightoutput)
#prob_final_model.load_weights(weightoutput)
#prediction_distribution = prob_bnn_model(tf.convert_to_tensor(df_standard[selected_features].copy(), dtype=tf.float32))

#df["IsTest"] = df_standard["IsTest"]
# The 95% CI is computed as mean ± (1.96 * stdv)
#df['prediction_u_non-fail','prediction_ubound-fail'] = test['prediction_mean_fail'] + (1.96 * test['prediction_std_fail'])
#df['prediction_u_non-fail','prediction_ubound-fail'] = test['prediction_mean_fail'] - (1.96 * test['prediction_std_fail'])
#prediction_stdv = prediction_stdv.tolist()
# selected_output_features = selected_features.copy()#.tolist()

# selected_output_features.extend(FEATURE_OMISSION)
# #selected_output_features.append("prediction_mean_non-fail")
# #selected_output_features.append("prediction_std_non-fail")
# selected_output_features.append("prediction_std_fail")
# selected_output_features.append("prediction_mean_fail")
# selected_output_features.append("Prediction Mean Rounded")
# selected_output_features.append("Will or Won't fail")
# selected_output_features= selected_output_features[0:-1].copy()
df["Prediction Mean Rounded"] = round(df["prediction_mean_fail"],2)
df["Will or Won't fail"] = df[failed_banks.columns[5]]<"2049-12-31"
mask = df['REPDTE'] > pd.to_datetime('2020-01-01')
#df[selected_output_features][mask].to_csv(rf"/mnt/c/Users/Wyatt Schwanbeck/source/repos/Test_2022-2023-RobustScaler-fold-best.csv")
  #time.sleep(5)



    # if(rmse<prior_rmse):
    #       prob_bnn_model.save_weights(r"/mnt/c/Users/Wyatt Schwanbeck/source/repos/temp_training3")
    #       time.sleep(5)
    #       #backupModel= keras.models.clone_model(prob_bnn_model)
    #       prior_rmse = rmse
    # else:
    #       prob_bnn_model.load_weights(r"/mnt/c/Users/Wyatt Schwanbeck/source/repos/temp_training3")
    # # df_standard.loc[val_index, 'IsTest']= True
#df_standard['IsTest'].fillna(False, inplace=True)

# test_flag= []
# for index in df.index:
#     if index in val_indexes:
#         test_flag.append(True)
#     else:
#         test_flag.append(False)
        
#df['IsTest'] = test_flag
#train_data_fold = tf.data.Dataset.from_tensor_slices((np.expand_dims(train[selected_features].copy(),axis=0), np.expand_dims(train["Y"].astype(float).copy(),axis=0)))
#val_data_fold = tf.data.Dataset.from_tensor_slices((np.expand_dims(test[selected_features].copy(),axis=0), np.expand_dims(test["Y"].astype(float).copy(),axis=0)))
#run_experiment(prob_bnn_model, negative_loglikelihood, train_data_fold, val_data_fold, num_epochs=num_epochs)


#prob_bnn_model.load_weights(r"/mnt/c/Users/Wyatt Schwanbeck/source/repos/temp_training_improvement")
#prediction_distribution = prob_bnn_model(test_input)


# test['y_pred'] = y_pred

# # Get the false negatives
#false_negatives = test[(test['Y'] == 1) & (test['y_pred'] == 0)]
# true_positives = test[(test['Y'] == 1) & (test['y_pred'] == 1)]
# false_positives = test[(test['Y'] == 0) & (test['y_pred'] == 1)]

# # Analyze false negatives by 'BankId' and 'VariableDate'
# grouped = false_negatives.groupby(['CERT', 'REPDTE']).size().reset_index(name='counts')
# grouped_true = true_positives.groupby(['CERT', 'REPDTE']).size().reset_index(name='counts')
# # Print the grouped DataFrame
# print(grouped)
# # Print out the accuracy score and confusion matrix
# print("Accuracy:", accuracy_score(y_test, y_pred))
# # Calculate and plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted:No', 'Predicted:Yes'], yticklabels=['Actual:No', 'Actual:Yes'])
# plt.show()


# # Perform predictions with uncertainty
# num_samples = 100  # Number of Monte Carlo samples
# predictions = []

# for _ in range(num_samples):
#     y_pred = prob_bnn_model.predict(X_test)
#     predictions.append(y_pred)

# # Compute uncertainty metrics (e.g., entropy, predictive variance)
# predictions = tf.concat(predictions, axis=0)
# entropy = -tf.reduce_mean(predictions * tf.math.log(predictions + 1e-8), axis=0)
# variance = tf.math.reduce_variance(predictions, axis=0)

# # Compute classification metrics (e.g., accuracy)
# predicted_class = tf.reduce_mean(predictions, axis=0) >= 0.5
# accuracy = tf.reduce_mean(tf.cast(predicted_class, tf.float32))

# print("Entropy:", entropy)
# print("Variance:", variance)
# print("Accuracy:", accuracy)