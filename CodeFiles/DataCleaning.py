# === libraries === 
import pandas as pa
import numpy as np

"""
Description: 
after reading the file we replace invalid values(0) with NaN,
then we replace the NaN values with the means of each feature.
"""

# === read raw dataset ===
df_train = pa.read_csv("DatasetZiad.csv")
df_test = pa.read_csv("TestZiad.csv")

# BMI cleaning data
col_bmi_train = df_train['BMI']
col_bmi_test = df_test['BMI']
col_bmi_train = col_bmi_train.replace(0, np.NaN)
col_bmi_test = col_bmi_test.replace(0, np.NaN)
col_bmi_train = col_bmi_train.replace(np.NaN, col_bmi_train.mean())
col_bmi_test = col_bmi_test.replace(np.NaN, col_bmi_test.mean())
df_train['BMI'] = col_bmi_train
df_test['BMI'] = col_bmi_test

# Blood Pressure data cleaning
col_bp_train = df_train['BloodPressure']
col_bp_test = df_test['BloodPressure']
col_bp_train = col_bp_train.replace(0, np.NaN)
col_bp_test = col_bp_test.replace(0, np.NaN)
col_bp_train = col_bp_train.replace(np.NaN, col_bp_train.mean())
col_bp_test = col_bp_test.replace(np.NaN, col_bp_test.mean())
df_train['BloodPressure'] = col_bp_train
df_test['BloodPressure'] = col_bp_test

# Skin Thickness data cleaning
col_sk_train = df_train['SkinThickness']
col_sk_test = df_test['SkinThickness']
col_sk_train = col_sk_train.replace(0, np.NaN)
col_sk_test = col_sk_test.replace(0, np.NaN)
col_sk_train = col_sk_train.replace(np.NaN, col_sk_train.mean())
col_sk_test = col_sk_test.replace(np.NaN, col_sk_test.mean())
df_train['SkinThickness'] = col_sk_train
df_test['SkinThickness'] = col_sk_test

# Glucose cleaning data
col_glucose_train = df_train['Glucose']
col_glucose_test = df_test['Glucose']
col_glucose_train = col_glucose_train.replace(0, np.NaN)
col_glucose_test = col_glucose_test.replace(0, np.NaN)
col_glucose_train = col_glucose_train.replace(np.NaN, col_glucose_train.mean())
col_glucose_test = col_glucose_test.replace(np.NaN, col_glucose_test.mean())
df_train['Glucose'] = col_glucose_train
df_test['Glucose'] = col_glucose_test

# === write cleaned data ===
df_train.to_csv('CleanDataTrain.csv')
df_test.to_csv('CleanDataTest.csv')
