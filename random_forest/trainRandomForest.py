# Set local path
import sys
sys.path.append('random-forest')
sys.path.append('utils')


# Imports
import pandas as pd
import numpy as np
# import scipy.io as spio
from os.path import join
# from math import ceil
from utils.mat_tools import matfile_struct_to_dict as mat2dict
# from utils.mat_tools import extrapVentRadius
# import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import Standard0Scaler
# from sklearn.pipeline import make_pipeline

# from sklearn.preprocessing import minmax_scale
import random_forest.rf_train as rft
# from sklearn.neighbors import NearestNeighbors
# from scipy.spatial import KDTree
# import seaborn as sns
# import matplotlib.pyplot as plt

##### INPUT #####
# ---- Directories and files ----
data_dir = 'data/'

monte_carlo_fixed_vars_file = join(data_dir, 'KatlaHydro_v8_noLd_2024-06-30_N20000_fixed_MC_vars.mat')  # Simulation, fixed input parameters
monte_carlo_rand_vars_file  = join(data_dir, 'KatlaHydro_v8_noLd_2024-06-30_N20000_rand_MC_vars.parquet') # Simulation, randomized input parameters
 # I/O data table for main parameters of interest, water depths up to 360 m
monte_carlo_data_file_20k   = join(data_dir, 'KatlaHydro_v8_noLd_2024-06-30_N20000.parquet') 
 # data table for main parameters, scaled/non-dimensionalized
monte_carlo_scaled_file     = join(data_dir, 'KatlaHydro_v8_noLd_2024-06-30_N20000_scaled.parquet')
# Smaller test data set with water depths up to 500 m for testing
monte_carlo_data_file_10k   = join(data_dir, 'KatlaHydro_v8_noLd_2024-06-22_N10000.parquet') 
# Scaled test data set, mostly handy for validation and plotting
scaled_test_data_file       = join(data_dir, 'KatlaHydro_v8_noLd_2024-06-22_N10000_scaled.parquet') 

# ---- DATA PARAMS ----
search_radius = 0.05

# ---- RF PARAMS ----
# Split siize
train_size = 0.2 # Small set for now
test_size  = 0.1

n_estimators = 100
max_depth    = 20
max_samples  = 0.2
bootstrap    = True
oob_score    = True
n_jobs       = 4
min_samples_split = 5
min_samples_leaf = 4
verbose = 1

# -----Grid search params -----
# -> first search
# param_grid = {'n_estimators' : [10, 20, 50],
#               'max_depth' :  [None,5,20,50],
#               # 'min_samples_split' : [5],
#                 # 'min_samples_leaf' : [1,2],
#                 'max_samples' : [None, 0.1, 0.2, 0.4],
#               }

# -> refined search
# param_grid = {'n_estimators' : [20,30, 50, 100, 200],
#                'max_depth' :  [None,20],
#               # 'min_samples_split' : [5],
#                 # 'min_samples_leaf' : [1,2],
#                 'max_samples' : [None, 0.2, 0.3, 0.4, 0.5, 0.6],
#               }
param_grid = {'n_estimators' : [200],
               'max_depth' :  [None],
              # 'min_samples_split' : [5],
                # 'min_samples_leaf' : [1,2],
                'max_samples' : [None, 0.4, 0.6],
              }
# Which estimators to evaluate separately in usage simulation
top_n_estimators_to_test = 75 

# Set of SCALED 2D coordinate points for creating radius-averaged point clouds
Ze_range_train = (0, 360)  # Water depth range in training set
Ze_range = (0, 500)  # Water depth range in test set
Q_range  = (6, 8)    # LogQ range in training set

# Simulation parameters testing a hypothetical usage case for the random forest predictons
Q_vector = [6e6, 4e7] # Q, kg/s - Span about an order of magnitude in mass flux for testing
Ze_max = [50, 150] #, 180] # sample MAX water depth occurring in each simulation
search_radius = 0.07


############################### DO THE THING ##################################

# Load up data
print('Loading data...')
monte_carlo_fixed_vars = mat2dict(monte_carlo_fixed_vars_file,'fixedVars')
monte_carlo_rand_vars  = pd.read_parquet(monte_carlo_rand_vars_file)
mc_df                  = pd.read_parquet(monte_carlo_data_file_20k)
mc_df_scaled           = pd.read_parquet(monte_carlo_scaled_file)
mc_df_10k             = pd.read_parquet(monte_carlo_data_file_10k)
mc_df_10k_scaled      = pd.read_parquet(scaled_test_data_file)

# ----- Some data prep and labeling -------
# Build some labels for later use
input_vars = ['Ze','logQ'] #,'n_total','a_over_Rv','T'] # monte_carlo_rand_vars['Variable']
scaled_input_vars = ['Ze_over_Rv','logQ']

output_vars = ['hm','rC','qw0','qwC','qs0','qsC']
scaled_output_vars = ['hm_over_Q14','rC_over_Rv','qw0_over_Q','qwC_over_Q','qs0_over_Q','qsC_over_Q']


# Fill some nans in the second set that weren't previously dealt with
mc_df_10k['hm'] = mc_df_10k['hm'].fillna(0.)

# Concatenate the two data sets
mc_df = pd.concat([mc_df, mc_df_10k],axis=0,ignore_index=True)
mc_df_scaled = pd.concat([mc_df_scaled, mc_df_10k_scaled],axis=0,ignore_index=True)

# Will use the log for training and testing
mc_df['logQ'] =  np.log10(mc_df['Q'])  # --> fully compiled monte carlo dataframe

# ---- Build smoothed train and test sets ----

averaged_data_set = rft.get_averaged_point_cloud(mc_df[input_vars+output_vars],
                                                 coordinate_columns=['Ze','logQ'],
                                                 scale_ranges = {'Ze':Ze_range_train, 'logQ': Q_range},
                                                 radius = search_radius)


# Get non-dimensionalized version
averaged_scaled_data = rft.apply_physical_data_scaling(averaged_data_set, mc_df['Q'])

# ---- Build classifier and train initial random forest ----
rf_kwargs = {'n_estimators' : n_estimators,
         # 'max_depth' : 20,
         'bootstrap' : bootstrap, 
         'oob_score' : oob_score,
         'n_jobs' : n_jobs,
         # 'max_samples' : max_samples,
         # 'min_samples_split' : min_samples_split,
         # 'min_samples_leaf' : min_samples_leaf,
         'random_state' : 0,
         'verbose' : verbose,
            }

regressor_rf = RandomForestRegressor(**rf_kwargs)

# ---- CHOOSE training, validation, test data ------

# It's useful to get a split of not just the key input vars, but also a couple other diagnostic vars:
#   e.g. collapse regime, non-dim Ze, etc
X_temp = pd.concat([mc_df[input_vars], mc_df_scaled[['Ze_over_Rv', 'clps_regime','Q']]], axis=1)

# New train/test set using only the 2 keys vars we care about
X_train, X_test, Y_train, Y_test = train_test_split(
    X_temp, averaged_data_set, test_size=0.2, random_state=42)

# Separate out training vars from housekeeping vars
X_train_extras = X_train[['Ze_over_Rv', 'clps_regime','Q']]
X_test_extras = X_test[['Ze_over_Rv', 'clps_regime','Q']]
X_train = X_train[input_vars]
X_test  = X_test[input_vars]



# ---- RUN USAGE CASE SIMULATION ------

# print('Running model sim...')
# normalized_mean_error, normalized_roughness_difference = rft.simulate_model_scenario(
#                             regressor_rf, 
#                             mc_df_test[input_vars + output_vars],
#                             Q = np.round(np.log10(Q_vector),decimals=1),
#                             Ze_max = Ze_max,
#                             coord_ranges = {'Ze': Ze_range, 'logQ': Q_range},
#                             search_radius = search_radius
#                            )

# print(normalized_mean_error)
# print(normalized_roughness_difference)
# print('Mean error: ',normalized_mean_error.mean(axis=None))
# print('Mean roughness diff: ',normalized_roughness_difference.mean(axis=None))


    

# ---- RUN GRID SEARCH OPTIMIZATION ------

grid_search = GridSearchCV(estimator=regressor_rf,
                           param_grid=param_grid,
                           n_jobs = n_jobs,
                           return_train_score=True,
                           scoring='neg_mean_squared_error',
                           verbose = 2)
grid_search.fit(X_train,Y_train)
results = pd.DataFrame(grid_search.cv_results_)

# Training parameters to have a look at
results_vars = ['rank_test_score',
                'mean_test_score',
                'std_test_score',
                'mean_train_score',
                'std_train_score',
                'mean_score_time']

results_vars = results_vars + ['param_' + par for par in param_grid.keys()]

key_results = results[results_vars].sort_values('rank_test_score',ignore_index=True)
print(key_results)


# ---- RUN USAGE CASE SIMULATIONS FOR BEST RESULTS ------
top_n_estimators_to_test = np.min([key_results.shape[0], top_n_estimators_to_test])

test_mse = np.zeros((top_n_estimators_to_test,1))
sim_nmse = np.zeros((top_n_estimators_to_test,1))
sim_nsse = np.zeros((top_n_estimators_to_test,1))
sim_mean_roughness_diff = np.zeros((top_n_estimators_to_test,1))
sim_std_roughness_diff = np.zeros((top_n_estimators_to_test,1))
normalized_mean_errors = []
normalized_roughness_differences = []
regressor_list = []

for ri in np.arange(0,top_n_estimators_to_test):
    rf_kw_ri = rf_kwargs.copy()
    for par in param_grid.keys():
        rf_kw_ri[par] = key_results.loc[ri,'param_' + par]
        
    regressor_list.append(RandomForestRegressor(**rf_kw_ri))
    regressor_list[ri].fit(X_train,Y_train)
    test_mse[ri] = metrics.mean_squared_error(Y_test, regressor_list[ri].predict(X_test))
    nmse, nsse, roughness_diff, roughness_diff_std = rft.simulate_model_scenario(
                                regressor_list[ri], 
                                mc_df[input_vars + output_vars],
                                Q = np.round(np.log10(Q_vector),decimals=1),
                                Ze_max = Ze_max,
                                coord_ranges = {'Ze': Ze_range, 'logQ': Q_range},
                                search_radius = search_radius,
                                fig_name = f'Result Rank: {key_results.loc[ri,"rank_test_score"]}',
                                )
    sim_nmse[ri] = nmse.mean(axis=None)
    sim_nsse[ri] = nsse.mean(axis=None)
    sim_mean_roughness_diff[ri] = roughness_diff.mean(axis=None)
    sim_std_roughness_diff[ri] = roughness_diff_std.mean(axis=None)
    
best_results = key_results.iloc[0:top_n_estimators_to_test]
best_results = best_results.assign(test_mse=test_mse)
best_results = best_results.assign(sim_nmse=sim_nmse)
best_results = best_results.assign(sim_nsse=sim_nsse)
best_results = best_results.assign(sim_mean_rough_diff=sim_mean_roughness_diff)
best_results = best_results.assign(sim_std_roughness_diff=sim_std_roughness_diff)

# -------- PLOT VALIDATION AND TEST SCORE RESULTS ---------

    
# rft.plot_RF_scores(best_results)
rft.plot_RF_scores(best_results, savefigs=True, descriptor='GridSearch2')


