# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:34:26 2023

@author: Wyatt Schwanbeck
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
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

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import RandomSearch, BayesianOptimization

from sklearn.metrics import mean_absolute_error, mean_squared_error

from CONFIG_MODEL_TRAINING import TARGETFILE, PROJECT_NAME, \
    calculate_precision_recall_fMeasure,show_std_dev_plot, \
        is_within_12_months_regression, \
        calculate_average_sampled_precision_recall, \
        df, FEATURE_OMISSION, failed_banks,selected_features

def build_model(hp):
    model = keras.Sequential()
    
    # Tune the number of units in the first Dense layer
    hp_units = hp.Int('units', min_value=128, max_value=512, step=8)
    model.add(layers.Dense(units=hp_units, activation='sigmoid'))
    
    # Tune the dropout rate
    hp_dropout_rate = hp.Float('dropout', min_value=0.01, max_value=0.10, step=0.01)
    model.add(layers.Dropout(rate=hp_dropout_rate))
    
    hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=16)
    model.add(layers.Dense(units=hp_units2, activation='sigmoid'))
    
    # Tune the dropout rate
    hp_dropout_rate2 = hp.Float('dropout2', min_value=0.01, max_value=0.25, step=0.01)
    model.add(layers.Dropout(rate=hp_dropout_rate2))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))  # sigmoid for output between 0 and 1
    
    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=hp_learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    
    return model

tuner = BayesianOptimization(
    build_model,
    objective='val_mean_squared_error',
    max_trials=50,  # set this to a higher value for more exhaustive search
    executions_per_trial=1,
    directory='dense-bank-failure-nn',
    project_name= PROJECT_NAME)

from sklearn.preprocessing import StandardScaler
import numpy as np

sections = []

for repdte in df["REPDTE"].unique():
    scaler = StandardScaler()
    df_section = df[df["REPDTE"]==repdte].copy()
    df_section[df.drop(FEATURE_OMISSION,axis=1).columns] = scaler.fit_transform(df_section.drop(FEATURE_OMISSION, axis=1).copy())
    sections.append(df_section.copy())
    
df_standard = pd.concat(sections,ignore_index=True) #df.copy() #
#df_standard = shuffle(df_standard)

df_standard['REPDTE'] = pd.to_datetime(df_standard['REPDTE'])
test = df_standard[df_standard['REPDTE'] >= '2022-09-30'][df_standard['REPDTE'] <= '2023-03-31']

#test.to_csv(r"/mnt/c/Users/Wyatt Schwanbeck/OneDrive/Documents/CSCI 794 - Masters Thesis/Bank Failure/TestDataWhole.csv")
train = df_standard[df_standard['REPDTE'] < '2022-09-30'][df_standard['REPDTE'] > '1992-01-01' ].copy()


# Split the dataset into train and test sets
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
import pandas as pd

X_train = train.drop(FEATURE_OMISSION, axis=1)

#X_train.dropna(inplace=True)
#X_train = train[features]
y_train = train['Y']

X_test = test.drop(FEATURE_OMISSION,axis=1)
y_test = test['Y']
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
# Set the number of folds and create a StratifiedKFold object
#rf = RandomForestRegressor(n_jobs=-1)
balanced_set = train[train["Y"] >0 ].copy()

balanced_set_Y = train[train["Y"]>0].copy()
balanced_test_Y = test[test["Y"]>0].copy()

balanced_set_test = pd.concat([test[test["Y"]==0].sample(len(balanced_test_Y), random_state=0), balanced_test_Y],ignore_index=True)

balanced_set = pd.concat([train[train["Y"]==0].sample(len(balanced_test_Y), random_state=0), balanced_test_Y],ignore_index=True)
balanced_sample_set = balanced_set.sample(int(len(balanced_set)*.2), random_state=42)
balanced_set.drop(balanced_sample_set.index)

tuner.search(balanced_sample_set[selected_features], balanced_sample_set["Y"],
             epochs=5,
             validation_data=(balanced_set_test[selected_features], balanced_set_test["Y"]))

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
model = tuner.hypermodel.build(best_hps)

import numpy as np


import pandas as pd

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

dataset_size = len(test)
batch_size = 128
#train_size = int(dataset_size * 0.85)


# And then fit and transform your data:
scaled_features_test = test[selected_features].copy()
scaled_features_balanced = balanced_set[selected_features].values

num_epochs = 11

features = train[selected_features].to_numpy()
labels = train["Y"].to_numpy()

# Convert labels to one-hot encoding if necessary
#one_hot_labels = tf.keras.utils.to_categorical(labels)
num_folds = 50
skf = StratifiedKFold(n_splits=num_folds)#, shuffle=True, random_state=43)
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
from tensorflow.python.ops import math_ops


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
# def mean_absolute_error(targets, estimated_distribution):
#     return math_ops.abs(targets - estimated_distribution)
from sklearn.utils import resample


df_std_temp= df_standard.copy()
test_input = df_std_temp[selected_features]
features = balanced_set[selected_features]
features.reset_index(drop=True)
labels = balanced_set["Y"] #train[selected_features].copy()#
labels.reset_index(drop=True)

min_mae = 1000
for fold in range(num_folds):
    print(f"Training fold: {fold+1}/{num_folds}")
    
    # Get the training and validation data for the current fold
    
    balanced_set = pd.concat([train[train["Y"]==False].sample(len(balanced_set_Y), random_state=fold), balanced_set_Y.sample(int(len(balanced_set_Y)), random_state=fold)],ignore_index=True)
    balanced_sample_set = balanced_set.sample(int(len(balanced_set)*.2), random_state=fold)
    balanced_set.drop(balanced_sample_set.index)
    features = balanced_set[selected_features]
    labels = balanced_set["Y"]
    x_train, x_val = balanced_set[selected_features], balanced_sample_set[selected_features]
    y_train, y_val = balanced_set["Y"], balanced_sample_set["Y"]
    # Get the training and validation data for the current fold
    # x_train, x_val = features.iloc[train_index], features.iloc[val_index]
    
    # y_train, y_val = labels.iloc[train_index].copy().astype(float), labels.iloc[val_index].astype(float)
    
    # x_train, y_train = resample(x_train,y_train, n_samples=1000)
    # x_val, y_val = resample(x_val,y_val, n_samples=1000)
    #train_data_fold = tf.data.Dataset.from_tensor_slices((np.expand_dims(x_train,axis=0), np.expand_dims(y_train,axis=0)))
    #val_data_fold = tf.data.Dataset.from_tensor_slices((np.expand_dims(x_val,axis=0), np.expand_dims(y_val,axis=0)))
    #train_data_fold = tf.data.Dataset.from_tensor_slices((x_train.copy(), y_train.copy()))
    #val_data_fold =  tf.data.Dataset.from_tensor_slices((x_val.copy(), y_val.copy()))
    #val_indexes.extend(val_index)
    print("Size x_train for fold: {0}".format(len(x_train)))
    #rmse = run_experiment(prob_bnn_model, negative_loglikelihood, train_data_fold, val_data_fold, num_epochs=num_epochs)
    #rf.fit(x_train,y_train)
    #test_df = df_standard[df_standard['REPDTE'] >= '2022-09-30']
    #history = model.fit(balanced_set[selected_features], balanced_set["Y"], epochs=1, validation_data=(test_df[selected_features], test_df["Y"]))
    
    history = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val))
    
    prediction_distribution = model.predict(test_input)
    #df_standard[['prediction_mean_fail']] = prediction_distribution
    #df_standard[['prediction_std_fail']] = prediction_distribution
    
    
    df_standard[['prediction_std_fail']] = prediction_distribution
    df_standard[['prediction_mean_fail']] = prediction_distribution
    # Splitting dataframe into train and test
    train_df = df_standard[df_standard['REPDTE'] < '2022-09-30']
    test_df = df_standard[df_standard['REPDTE'] >= '2022-09-30'][df_standard['REPDTE'] <= '2023-03-31']
    #df_standard[df_standard['REPDTE'] >= '2022-09-30'][df_standard['REPDTE'] <= '2023-03-31']['prediction_mean_fail'] = prediction_distribution
    # Compute RMSE for train and test
    # train_rmse = compute_rmse(train_df['prediction_mean_fail'], train_df['Y'])
    train_rmse = mean_absolute_error(train_df['prediction_mean_fail'],train_df['Y'],)
    #test_mae = mean_absolute_error(test_df[test_df['Y']>0]['prediction_mean_fail'],test_df[test_df['Y']>0]['Y'])
    #test_mae_nonFailed = mean_absolute_error(test_df[test_df['Y']==0]['prediction_mean_fail'],test_df[test_df['Y']==0]['Y'])
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

    # it's a good practice to use `.loc` for conditional subsetting
    for combo in bkclass_specgrp:
        #fig, ax = plt.subplots(nrows=3)
        if(combo[0] == "*"):
            #TrueFailPred = df_standard.loc[(df_standard["Y"] == True) & (df_standard["REPDTE"] == testDate), "prediction_mean_fail"]
            #NonFailPred = df_standard.loc[(df_standard["Y"] == False) & (df_standard["REPDTE"] == testDate), "prediction_mean_fail"]
            fig, axs = plt.subplots(nrows=3, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
            fig.suptitle(f'Bank Failure Predictions \n {PROJECT_NAME}- Artificial Neural Network - Epoch: {fold}', fontsize=12, fontweight='bold')
        else:
            
            #TrueFailPred = df_standard.loc[(df_standard["Y"] == True) & (df_standard["REPDTE"] == testDate) & (df_standard['BKCLASS'] == combo[0]) & (df_standard['SPECGRPN'] == combo[1]), "prediction_mean_fail"]
            #NonFailPred = df_standard.loc[(df_standard["Y"] == False) & (df_standard["REPDTE"] == testDate) & (df_standard['BKCLASS'] == combo[0]) & (df_standard['SPECGRPN'] == combo[1]), "prediction_mean_fail"]
            fig, axs = plt.subplots(nrows=2, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
            fig.suptitle(f'Bank Failure Predictions - {combo[0]} - {combo[1]} ', fontsize=12, fontweight='bold')
        
        for t, testDate in enumerate(combo[2]):
            if(combo[0] == "*"):
                TrueFailPred = df_standard.loc[(df_standard["Y"] > 0) & (df_standard["REPDTE"] == testDate), "prediction_mean_fail"]
                NonFailPred = df_standard.loc[(df_standard["Y"] == 0) & (df_standard["REPDTE"] == testDate), "prediction_mean_fail"]
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
                    axs[t].annotate(f"{int(count)}", (bin_start, count), textcoords="offset points", xytext=(5,5), ha='center', fontsize=6)
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
                        
                        maskAnnotate = (df_standard["Y"] > 0) & (df_standard["REPDTE"] == testDate) & (df_standard["BKCLASS"]==combo[0]) & (df_standard['SPECGRPN'] == combo[1]) & (df_standard["prediction_mean_fail"] >= bin_start) & (df_standard["prediction_mean_fail"] < bin_start + (bins[1] - bins[0]))
                    institutions = df_standard.loc[maskAnnotate, "NAME"]
                    
                    # Annotate the names, carefully adjusting positions to avoid clutter
                    # for i, institution in enumerate(institutions):
                    #     axs[t].annotate(institution, (bin_start, count * (1 + 0.1 * (i+1))), textcoords="offset points", xytext=(5,5), ha='center', fontsize=6, color='red')
                    # Annotate the names, carefully adjusting positions to avoid clutter
                    for i, institution in enumerate(institutions):
                        axs[t].annotate(institution, 
                                        (bin_start, count * (1 + 0.2 * (i+1))), 
                                        textcoords="offset points", 
                                        xytext=(5, (e%4)*25 + (i*25)), fontweight='bold',#xytext=(5 *(i+1), 10 + (10*bin_start) * (i+10+bin_start)), fontweight='bold',
                                        ha='center', 
                                        fontsize=8, 
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
        #if(test_mae<.1):
        plt.show()
        #plt.show()
        
    df[['prediction_mean_fail']] = prediction_distribution
    #df[['prediction_std_fail']] = prediction_distribution.stddev().numpy()
    df["Prediction Mean Rounded"] = round(df["prediction_mean_fail"],2)
    df["Will or Won't fail"] = df[failed_banks.columns[5]]<"2049-12-31"
    #time.sleep(5)
    # Add a title to the entire plot
    TrueFailPred = df_standard.loc[(df_standard["Y"] > 0) & (df_standard["REPDTE"] >= "9-30-2022"), "prediction_mean_fail"]
    NonFailPred = df_standard.loc[(df_standard["Y"] == 0) & (df_standard["REPDTE"] >= "9-30-2022"), "prediction_mean_fail"]
    if(NonFailPred.median()-TrueFailPred.median()<median_spread):
          median_spread = NonFailPred.median()-TrueFailPred.median()
  