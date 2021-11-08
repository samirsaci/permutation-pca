import pandas as pd
import numpy as np
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from pathlib import Path
from utils.train.train import (
    train_test)
from utils.train.importances import (
    features_importance
)
from utils.permutation.permutation import permutation

warnings.filterwarnings('ignore')

# settings to display all columns
pd.set_option("display.max_columns", None)

# Import Data
data_size = 'light'
data = pd.read_pickle('data/data_features_{}.pkl'.format(data_size))
# Start records from 55th
start_row = 55
data = data[data['d'] >= start_row].copy()
print("{:,} records for the prepared data".format(len(data)))
# Drop columns
data.drop(['Unnamed: 0'], axis = 1, inplace = True)

# Features columns
LIST_NAME = ['d_id', 'd_item_id', 'd_dept_id', 'd_cat_id', 'd_store_id', 'd_state_id']
dict_data = {}
for list_name in LIST_NAME:
    dict_temp = pickle.load(open('data/params/{}.p'.format(list_name), "rb"))
    dict_data[list_name] = dict_temp
    del dict_temp

# Initial Features
INIT_FEAT = list(data.columns[0:21])
# Lags and averages
LAGAV_FEAT = list(data.columns[24:42])
# Rolling Means and Rolling Means on lag
ROLLMEAN_FEAT = list(data.columns[42:52])
# Trends and Rolling MAX
TREND_MAX_FEAT = list(data.columns[52:58])
# Stock-Out and Store Closed
# SO_CLOSED_FEAT = list(data.columns[58:67])
SO_CLOSED_FEAT = list(['stock_out_id', 'store_closed'])
# PRICE COMPARISON
PRICE_COMPARE = list(data.columns[21:24])
# Dictionnary with different steps
dict_features = {
    'STEP_1': INIT_FEAT,
    'STEP_2': INIT_FEAT+LAGAV_FEAT,
    'STEP_3': INIT_FEAT+LAGAV_FEAT+ROLLMEAN_FEAT,
    'STEP_4': INIT_FEAT+LAGAV_FEAT+ROLLMEAN_FEAT+TREND_MAX_FEAT,
    'STEP_5': INIT_FEAT+LAGAV_FEAT+ROLLMEAN_FEAT+TREND_MAX_FEAT+SO_CLOSED_FEAT,
    'STEP_6': INIT_FEAT+LAGAV_FEAT+ROLLMEAN_FEAT+TREND_MAX_FEAT+SO_CLOSED_FEAT+PRICE_COMPARE,
}
LIST_STEPS = ['STEP_1', 'STEP_2', 'STEP_3', 'STEP_4', 'STEP_5', 'STEP_6']
LIST_STEPS_NAME = ['INITIAL_DATA', 'INITIAL + LAG + AVERAGES', 
                  'INITIAL + LAG + AVERAGES + ROLLING MEAN',
                  'INITIAL + LAG AND AVERAGES + ROLLING MEAN + TREND AND ROLLING MAX',
                  'INITIAL + LAG AND AVERAGES + ROLLING MEAN + TREND AND ROLLING MAX + STOCK-OUT AND STORE CLOSED',
                  'INITIAL + LAG AND AVERAGES + ROLLING MEAN + TREND AND ROLLING MAX + STOCK-OUT AND STORE CLOSED + PRICE COMPARISON']
dict_stepname = dict(zip(LIST_STEPS, LIST_STEPS_NAME))

### BASELINE WITH ALL FEATURES
## TRAINING, TEST 
folder_name = 'models/initial/'
dict_error = {}
model_name = 'initial'
# Get the store ids
features = dict_features['STEP_6']
stores = data.store_id.unique()
d_store_id = dict_data['d_store_id']
# Train your model
df_validpred = train_test(data, features, stores, d_store_id, folder_name, model_name)
# Compute Error
valid_rmse = 100 * np.sqrt(np.mean((df_validpred.validation.values - df_validpred.prediction.values) ** 2))/np.mean(df_validpred.validation.values)
# Add Error in a Dictionnary
dict_error['all_features'] = valid_rmse
print("{}: RMSE = {}".format('all_features', valid_rmse))
## PLOT FEATURES IMPORTANCE
n_features = 25
features_importance(folder_name, 'initial', features, d_store_id, valid_rmse, stores, n_features)

### PERMUTATION IMPORTANCE
## PERMUTATION LOOP
# Baseline with initial features
baseline_rmse = valid_rmse
print("Baseline with all features: RMSE = {}".format(baseline_rmse))
# Parameters
all_features = dict_features['STEP_6']
stores = data.store_id.unique()
d_store_id = dict_data['d_store_id']
folder_model = 'models/initial/'
# Dictionnary of error
dict_error = {}
dict_error['baseline'] = baseline_rmse
# Perform permutation
df_permres, df_validall, dict_error = permutation(data, all_features, stores, d_store_id, baseline_rmse, dict_error, folder_model)
df_permres.to_csv(folder_model + 'permutation_error' + '.csv')
# Plot
ax = plt.gca()
df_permres[df_permres['delta']<0].sort_values(['delta'], ascending = True).plot.barh(figsize=(8, 5), x='remove', y='delta', ax=ax, color = 'black')
plt.ylabel('Feature removed')
plt.xlabel('Δ RMS')
plt.xticks(rotation=90)
plt.title('List of features that reduce the error after being removed (permutation method)')
plt.show()
# Features to remove 
list_remove = list(df_permres[df_permres['delta']<0]['remove'].unique())
print("Features to be removed from the training set to improve the accuracy {}".format(list_remove))

### RETRAIN THE ALGORITHM WITH UPDATED THE FEATURES
## TRAIN ON A NEW SCOPE OF FEATURES
# Model name
model_name = 'permutation'
# Update the features
new_features = [feat for feat in all_features if (feat not in list_remove) or (feat in ['sold'])]
# Models with filtered scope
folder_name = 'models/permutation/'
# DataFrame with filter scope
data_scope = data.copy()
# Train with the filtered scope
df_validpred = train_test(data, new_features, stores, d_store_id, folder_name, model_name)
# Save Prediction for all stores
df_validpred.to_csv(folder_name + 'error_prediction_filtered.csv')
# Compute Error
valid_rmse = 100 * np.sqrt(np.mean((df_validpred.validation.values - df_validpred.prediction.values) ** 2))/np.mean(df_validpred.validation.values)
# Add Error in a Dictionnary
dict_error['all_features'] = valid_rmse
print("{}: RMSE = {}".format('filtered', valid_rmse))
## PLOT FEATURES IMPORTANCE
folder_model = 'models/permutation/'
n_features = 25
features_importance(folder_model, 'permutation', new_features, d_store_id, valid_rmse, stores, n_features)