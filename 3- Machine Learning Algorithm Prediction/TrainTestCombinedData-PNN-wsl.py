# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:34:26 2023

@author: Wyatt Schwanbeck
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import os
#from tensorflow import keras
import keras
from keras import layers

#Ensures rounding is consistent in TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load dataset
# Hide GPU from visible devices as PNN requires more memory than GPUs typically have
#tf.config.set_visible_devices([], 'GPU')

#Retreive target dataset based on config
from CONFIG_MODEL_TRAINING import TARGETFILE, PROJECT_NAME, \
    calculate_precision_recall_fMeasure,show_std_dev_plot, \
        is_within_12_months_regression, \
        calculate_average_sampled_precision_recall, \
        df, FEATURE_OMISSION, failed_banks, selected_features

from sklearn.preprocessing import StandardScaler


#COMPILE StandardScalar dataframe by report date
sections = []
for repdte in df["REPDTE"].unique():
    scaler = StandardScaler()
    df_section = df[df["REPDTE"]==repdte].copy()
    df_section[df.drop(FEATURE_OMISSION,axis=1).columns] = scaler.fit_transform(df_section.drop(FEATURE_OMISSION, axis=1).copy())
    sections.append(df_section.copy())
    
df_standard = pd.concat(sections,ignore_index=True)
df_standard['REPDTE'] = pd.to_datetime(df_standard['REPDTE'])

test = df_standard[df_standard['REPDTE'] >= '2022-09-30']
train = df_standard[df_standard['REPDTE'] < '2022-12-31'][df_standard['REPDTE'] > '2000-01-01' ].copy()

balanced_set_Y = train[train["Y"]>0].copy()
balanced_set = pd.concat([train[train["Y"]==0].sample(len(balanced_set_Y)), balanced_set_Y],ignore_index=True)


# Define variational posterior weight distribution as multivariate Normal Tril.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Laplace(#
                    loc=tf.zeros(n),scale=tf.ones(n)
                ))
            
        ]
    )
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [

            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n)
            
        ]
    )
    return  posterior_model


hidden_units = [64]#[len(selected_features),len(selected_features)*2,2]
learning_rate = 0.001
def run_experiment(model, loss, train_dataset, test_dataset,num_epochs=100):


    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    #print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    #print(f"Train rmse Divergence: {round(rmse, 3)}")

    #print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    #print(f"Test  rmse: {round(rmse, 3)}")
    return rmse

# And then fit and transform your data:
scaled_features_test = test[selected_features].copy()
scaled_features_balanced = balanced_set[selected_features].values

def create_probablistic_bnn_model(kl_weight=.0001):
    inputs = tf.keras.layers.Input(shape=len(selected_features),)
    
    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    hiddenLayers = 0
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=kl_weight,
            activation="sigmoid",
        )(inputs)
        
            
        hiddenLayers +=1
    features = layers.Dropout(.05)(features)
    
    distribution_params = layers.Dense(units=1)(features)
    outputs = tfp.layers.IndependentBernoulli(1,convert_to_tensor_fn=tfp.distributions.Bernoulli.logits)(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=negative_loglikelihood,
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model

from tensorflow.python.ops import math_ops


mse_loss = keras.losses.MeanSquaredError()

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

from sklearn.metrics import mean_absolute_error

num_epochs = 1

features = train[selected_features].to_numpy()
labels = train["Y"].to_numpy()

num_folds = 20

prior_pred = 0
median_spread = 100

selected_output_features = selected_features.copy()#.tolist()
selected_output_features.extend(FEATURE_OMISSION)
selected_output_features.append("prediction_std_fail")
selected_output_features.append("prediction_mean_fail")
selected_output_features.append("Prediction Mean Rounded")
selected_output_features.append("Will or Won't fail")
selected_output_features= selected_output_features[0:-1].copy()

mask = df['REPDTE'] > pd.to_datetime('2020-01-01')

bkclass_specgrp= [ ("*", "*", ["2023-06-30","2023-09-30", "2023-12-31"], "test")]

df_std_temp= df_standard[selected_features].copy()
test_input = tf.convert_to_tensor(df_std_temp[selected_features], dtype=tf.float32)

fold_min_mae = 0
sample_count = 5
for sampling in range(sample_count):
    prob_bnn_model = create_probablistic_bnn_model()
    prediction_distribution = prob_bnn_model(test_input, training=False)
    df_standard[['prediction_mean_fail']] = prediction_distribution.mean().numpy() #temp swap to stddev() from mean() to visualize certainty
    df_standard[['prediction_std_fail']] = prediction_distribution.stddev().numpy()
    test_df = df_standard[df_standard['REPDTE'] >= '2023-03-31'][df_standard['REPDTE'] <= '2024-03-31'][df_standard['BKCLASS'].isin(['NM',"SM","N"])]
    train_sample_df = df_standard[df_standard['REPDTE'] >= '2010-09-30'][df_standard['REPDTE'] <= '2010-03-31'][df_standard['BKCLASS'].isin(['NM',"SM","N"])]
    test_mae = mean_absolute_error(test_df[test_df['Y']>0]['prediction_mean_fail'],test_df[test_df['Y']>0]['Y'])

    test_mae_nonFailed = mean_absolute_error(test_df[test_df['Y']==0]['prediction_mean_fail'],test_df[test_df['Y']==0]['Y'])
    if(sampling==0):
        overall_precision = 0
        overall_recall = 0
        overall_fmeasure = 0
        max_fmeasure = overall_fmeasure #test_mae + test_mae_nonFailed
    for fold in range(num_folds):
        print(f"Training fold: {fold}/{num_folds}")
        
        # Get the training and validation data for the current fold
        balanced_set = pd.concat([train[train["Y"]==False].sample(len(balanced_set_Y)*10, random_state=fold), balanced_set_Y.sample(int(len(balanced_set_Y)),random_state=fold)],ignore_index=True)
        balanced_sample_set = balanced_set.sample(int(len(balanced_set)*.1), random_state=fold)
        balanced_set.drop(balanced_sample_set.index)
        balanced_set = balanced_set.sample(frac=1).reset_index(drop=True)
        features = balanced_set[selected_features]
        labels = balanced_set["Y"]
        x_train, x_val = balanced_set[selected_features], balanced_sample_set[selected_features]
        y_train, y_val = balanced_set["Y"], balanced_sample_set["Y"]

        train_data_fold = tf.data.Dataset.from_tensor_slices((np.expand_dims(x_train,axis=0), np.expand_dims(y_train,axis=0)))
        val_data_fold = tf.data.Dataset.from_tensor_slices((np.expand_dims(x_val,axis=0), np.expand_dims(y_val,axis=0)))

        #Train Model
        rmse = run_experiment(prob_bnn_model, negative_loglikelihood, train_data_fold, val_data_fold, num_epochs=num_epochs)
        
        #Test Model and determine performance between train and test
        prediction_distribution = prob_bnn_model(test_input, training=False)
        
        df_standard[['prediction_mean_fail']] = prediction_distribution.mean().numpy() #temp swap to stddev() from mean() to visualize certainty
        df_standard[['prediction_std_fail']] = prediction_distribution.stddev().numpy()
        
        # Splitting dataframe into train and test
        train_df = df_standard[df_standard['REPDTE'] < '2022-09-30'][df_standard['REPDTE'] <= '2023-03-31']
        test_df = df_standard[df_standard['REPDTE'] >= '2022-09-30'][df_standard['REPDTE'] <= '2024-03-31'] #[df_standard['BKCLASS'].isin(['NM',"SM"])]
        
        #Calculate MAE
        train_mae = mean_absolute_error(train_df['prediction_mean_fail'],train_df['Y'])
        test_mae = mean_absolute_error(test_df[test_df['Y']>0]['prediction_mean_fail'],test_df[test_df['Y']>0]['Y'])
        test_mae_nonFailed = mean_absolute_error(test_df[test_df['Y']==0]['prediction_mean_fail'],test_df[test_df['Y']==0]['Y'])
        
        
        print(f"Test MAE Non-Failed Banks : {round(test_mae_nonFailed,2)}  \t  Train MAE: {round(test_mae,2)}") 
        test_precision, test_recall, test_fmeasure = calculate_average_sampled_precision_recall(df_standard, "2023-03-31", "2024-03-31")
        train_precision, train_recall, train_fmeasure = calculate_average_sampled_precision_recall(df_standard, "2000-01-01", "2022-12-31")
        if(test_fmeasure+train_fmeasure> max_fmeasure):
             print(f"Test MAE Non-Failed Banks : {test_mae_nonFailed}")
             #print(f"Train MAE: {train_mae}")
             print(f"Test MAE: {test_mae}")
             max_fmeasure = test_fmeasure+train_fmeasure # test_mae +test_mae_nonFailed
             fold_min_mae = fold
             
             print(f"Test Precision: {round(test_precision,2)} \t Recall: {round(test_recall,2)} \t fmeasure: {round(test_fmeasure,2)} ")
             print(f"Train Precision: {round(train_precision,2)} \t Recall: {round(train_recall,2)} \t fmeasure: {round(train_fmeasure,2)} ")
             for combo in bkclass_specgrp:
                #fig, ax = plt.subplots(nrows=3)
                if(combo[0] == "*"):
                    if(combo[3]=="test"):
                        show_std_dev_plot(df_standard, combo[2])
                    rounded_failing_bank_mae = int(test_mae*100)
                    rounding_nonfailing_bank_mae = int(test_mae_nonFailed*100)
                    fig, axs = plt.subplots(nrows=3, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
                    fig.suptitle(f'Bank Failure Predictions \n {PROJECT_NAME} -' \
                            + ' Probabilisitic Neural Network \n ' \
                            + f'Failing Bank MAE: .{rounded_failing_bank_mae} ' \
                            + f'Non-Failing Bank MAE: .{rounding_nonfailing_bank_mae} '\
                            + f'k-fold {fold} \n \n ' \
                            + f'Test Precision: {round(test_precision,2)} Test Recall: {round(test_recall,2)} Test F-Measure: {round(test_fmeasure,2)} \n' \
                            + f'Train Precision: {round(train_precision,2)} Train Recall: {round(train_recall,2)} Train F-Measure: {round(train_fmeasure,2)}', fontsize=10, fontweight='bold')
                    
                elif(combo[0]== "*2"):
                    fig, axs = plt.subplots(nrows=3, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
                    fig.suptitle(f'Bank Failure Prediction Standard Deviations - k-fold {fold}', fontsize=12, fontweight='bold')
                else:
                    
                    fig, axs = plt.subplots(nrows=2, figsize=(10, 15))  # you might want to specify figsize to adjust the size of the whole plot
                    fig.suptitle(f'Bank Failure Predictions - {combo[0]} - {combo[1]} - k-fold {fold}', fontsize=12, fontweight='bold')
                
                for t, testDate in enumerate(combo[2]):
                    if(combo[0] == "*"):
                        colors = ['#E69F00', '#56B4E9']#, '#FF0000']
                        TrueFailPred = df_standard.loc[(df_standard["Y"] > 0) & (df_standard["REPDTE"] == testDate) & (df_standard["BKCLASS"].isin(["NM","SM"])), ["prediction_mean_fail","Y"]]
                        NonFailPred = df_standard.loc[(df_standard["Y"] == 0) & (df_standard["REPDTE"] == testDate) & (df_standard["BKCLASS"].isin(["NM","SM"])) , ["prediction_mean_fail","Y"]] 
                        TrueFailingBank_mae = mean_absolute_error( TrueFailPred["Y"], TrueFailPred["prediction_mean_fail"])
                        NonFailingBank_mae = mean_absolute_error(NonFailPred["Y"], NonFailPred["prediction_mean_fail"])
                        NonFailPred = NonFailPred["prediction_mean_fail"]
                        TrueFailPred = TrueFailPred["prediction_mean_fail"]

                    elif(combo[0] == "*2"):
                        colors = ['#AB400C', '#320000']#, '#FF0000']
                        TrueFailPred = df_standard.loc[(df_standard["Y"] > 0) & (df_standard["REPDTE"] == testDate) & (df_standard["BKCLASS"].isin(["NM","SM"])), "prediction_std_fail"]
                        NonFailPred = df_standard.loc[(df_standard["Y"] == 0) & (df_standard["REPDTE"] == testDate) & (df_standard["BKCLASS"].isin(["NM","SM"])), "prediction_std_fail"]
                    else:
                        colors = ['#E69F00', '#56B4E9']#, '#FF0000']
                        TrueFailPred = df_standard.loc[(df_standard["Y"] > 0) & (df_standard["REPDTE"] == testDate) & (df_standard['BKCLASS'] == combo[0]) & (df_standard['SPECGRPN'] == combo[1]), "prediction_mean_fail"] #(df_standard["CHANGECODE_DESC_LONG"]==0)
                        NonFailPred = df_standard.loc[(df_standard["Y"] == 0) & (df_standard["REPDTE"] == testDate) & (df_standard['BKCLASS'] == combo[0]) & (df_standard['SPECGRPN'] == combo[1]), "prediction_mean_fail"]

                    precision, recall, fmeasure = calculate_average_sampled_precision_recall(df_standard[df_standard["REPDTE"]==testDate], testDate,testDate)
                    names = [f"Non-Fail Bank Predictions", f"Actual Failed Bank Predictions"]  
                    
                    # Note that `axs[t]` is used to specify the subplot where the hist should be drawn
                    n, bins, patches = axs[t].hist([NonFailPred, TrueFailPred], label=names, color=colors, log=True, bins=40)
                    axs[t].legend()
                    axs[t].set_title(f"Count of Prediction Rates - {testDate} - \n Precision: {round(precision,2)}   Recall: {round(recall,2)}   F-Measure: {round(fmeasure,2)}")  # set title for each subplot
                    # Annotating the counts above the bars
                    bin_num = 0
                    for count, bin_start, patch in zip(n[0], bins, patches[0]):
                        if count > 0:
                            axs[t].annotate(f"{int(count)}", (bin_start, count), textcoords="offset points", xytext=(5,5), ha='center', fontsize=7)
                        bin_num+=1
                   
                    for e, zipped in enumerate(zip(n[1], bins, patches[1]),1):
                        count, bin_start, patch = zipped
                        if count > 0:
                            axs[t].annotate(f"{int(count)}", (bin_start, count), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
                            
                            # Get the names of the institutions for the TruePred values in the current bin 
                            if(combo[0] == "*" or combo[0] == "*2"):
                                
                                maskAnnotate = (df_standard["Y"] > 0) & (df_standard["REPDTE"] == testDate) & (df_standard["prediction_mean_fail"] >= bin_start) & (df_standard["prediction_mean_fail"] < bin_start + (bins[1] - bins[0]))
                            else:
                                
                                maskAnnotate = (df_standard["Y"] > 0) & (df_standard["REPDTE"] == testDate) & (df_standard["BKCLASS"]==combo[0]) & (df_standard['SPECGRPN'] == combo[1]) & (df_standard["prediction_mean_fail"] >= bin_start) & (df_standard["prediction_mean_fail"] < bin_start + (bins[1] - bins[0]))
                            institutions = df_standard.loc[maskAnnotate, "NAME"]
                            if(combo[3]=="test"):
                                # Annotate names to minimize clutter
                                for i, institution in enumerate(institutions):
                                    axs[t].annotate(institution, 
                                                    (bin_start, count * (1 + 0.2 * (i+1))), 
                                                    textcoords="offset points", 
                                                    xytext=(5, (e%4)*25 + (i*25)), fontweight='bold',
                                                    ha='center', 
                                                    fontsize=7, 
                                                    color='black')
                    
                    axs[t].xaxis.set_major_formatter(PercentFormatter(1))

                    # Draw the vertical line used to denote cut-off for classification metrics
                    axs[t].axvline(x=.50, color='red', linestyle='--')
        
                plt.tight_layout(pad=4.0)
                if(combo[0]=="*"):
                    plt.savefig(f'PNN_Figures/Prediction-Summary-k-fold-{fold}.png')

                plt.show()  
                df[['prediction_mean_fail']] = prediction_distribution.mean().numpy()
                df[['prediction_std_fail']] = prediction_distribution.stddev().numpy()
                df["Prediction Mean Rounded"] = round(df["prediction_mean_fail"],2)
                df["Will or Won't fail"] = df[failed_banks.columns[5]]<"2049-12-31"
                df[selected_output_features][mask].to_csv(rf"PNN_Test_Output/Test_2022-2023-RobustScaler-fold-{fold}.csv")

                # Add a title to the entire plot
                TrueFailPred = df_standard.loc[(df_standard["Y"] > 0) & (df_standard["REPDTE"] >= "9-30-2022"), "prediction_mean_fail"]
                NonFailPred = df_standard.loc[(df_standard["Y"] == 0) & (df_standard["REPDTE"] >= "9-30-2022"), "prediction_mean_fail"]

