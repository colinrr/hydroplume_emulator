#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:55:37 2024

@author: crrowell
"""

# Run a random forest regressor on Katla Monte Carlo ouput

import pandas as pd
import numpy as np
# import scipy.io as spio
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


data_dir = '/Users/crrowell/code/research-projects/katla/katlaPlumePhysicsEmulator/data'
monte_carlo_data_file       = join(data_dir, 'KatlaHydro_v8_noLd_2024-06-22_N10000.parquet')  # I/O data table for main parameters of interest


# ---- RF PARAMS ----
# Split siize
train_size = 0.2 # Small set for now
test_size  = 0.1

n_estimators = 100
max_depth    = 3
bootstrap    = True
oob_score    = True
n_jobs       = 2
min_samples_split = 2
# min_samples_leaf = 1
verbose = 1



####  ----  DO THE THING ---- ####
# Load up data frame
mc_df                  = pd.read_parquet(monte_carlo_data_file)
mc_df['hm'] = mc_df['hm'].fillna(0.) # Should do this before in post-processing...
               
# Get logQ
logQ = np.log10(mc_df['Q'])
logQ.name = 'logQ'

# Get total vapor series
n_total = mc_df['n_0'] + mc_df['n_ec']
n_total.name = 'n_v'

# Setup input/output variables - consider non-dimensionalizing, then normalizing
X = pd.concat([logQ, mc_df['Ze'], n_total], axis=1)
Y = mc_df[['hm', 'qs0', 'qsC', 'rC', 'qw0', 'qwC']]


# Train/Test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = train_size, test_size = test_size, random_state=42)
X_train.shape, X_test.shape

# Build classifier
regressor_rf = RandomForestRegressor(n_estimators=n_estimators, 
                                     max_depth=max_depth,
                                     bootstrap=bootstrap, 
                                     oob_score=oob_score,
                                     n_jobs=n_jobs,
                                     min_samples_split=min_samples_split,
                                     random_state=0,
                                     verbose=verbose)


#%% Run initial fit and make initial predictions


regressor_rf.fit(X,Y)

regressor_rf.oob_score_

predictions = regressor_rf.predict(X_test)

#%% Check feature importances

# making the feature importances plot
feature_imp = pd.Series(regressor_rf.feature_importances_, 
                        index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title("Visualizing Feature Importances", fontsize=15)

#%% Run a hyperparameter grid search

#%% 

