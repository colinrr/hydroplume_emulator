#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 08:46:16 2024

@author: crrowell
"""

from scipy.signal import savgol_filter as sgf

def get_roughness(y):
    if type(y)==pd.DataFrame:
        y_norm = (y - y.mean(axis=0)) / y.std(axis=0)
        mean_roughness = (y_norm.diff(axis=0).diff(axis=0)**2).mean(axis=0)
        std_roughness  = (y_norm.diff(axis=0).diff(axis=0)**2).std(axis=0)
    elif type(y)==pd.Series:
        y_norm = (y - y.mean()) / y.std()
        mean_roughness = (y_norm.diff().diff()**2).mean()
        std_roughness = (y_norm.diff().diff()**2).std()
    else:
        raise TypeError
    return mean_roughness, std_roughness

def get_roughness_difference(y_true,y_pred):
    assert np.shape(y_true)==np.shape(y_pred), 'Array sizes do not match.'
    if type(y_true)==pd.DataFrame and type(y_pred)==pd.DataFrame:
        assert np.all(y_true.columns==y_pred.columns)
        
    roughness_true = get_roughness(y_true)
    roughness_pred = get_roughness(y_pred)
    normalized_roughness_difference = roughness_pred - roughness_true
    return normalized_roughness_difference

y_predict = pd.DataFrame(data=y_predict,columns=output_vars)
# y_true = query_avg['qwC']

y = pd.concat([query_avg['qwC'], y_predict['qwC']], axis=1, keys = ['true', 'y'])

poly_list= [10,20]
rd = np.zeros((len(poly_list)+1))

for pl in poly_list:
    y = pd.concat([y, pd.Series(sgf(y_predict['qwC'], pl, 3),name=f'y_{pl}')], axis=1)
    
plt.plot(t,y)
print(get_roughness(y))

