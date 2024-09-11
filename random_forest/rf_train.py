#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:35:42 2024

@author: crrowell
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple
from math import ceil
from utils.mat_tools import extrapVentRadius
from os.path import join

"""
Various helper functions for training a random forest on Monte Carlo simulation data.
"""
plt.rcParams['text.usetex'] = True
var_labels = {'Ze' : r'Water Depth, $\displaystyle{Z_e}$ (m)',
             'Q'   : r'Source mass flux, $\displaystyle{Q}$ (kg/s)',
             'logQ': r'Log source mass flux, $\displaystyle{log_{10}(Q)}$',
             'hm'  : r'Plume height, $\displaystyle{h_m} (m)$',
             'rC'  : r'Collapse radius, $\displaystyle{r_C} (m)$',
             'qw0' : r'Surface water flux, $\displaystyle{q_{w0}} (kg/s)$',
             'qwC' : r'Collapse water flux, $\displaystyle{q_{wC}} (kg/s)$',
             'qs0' : r'Surface particle flux, $\displaystyle{q_{s0}} (kg/s)$',
             'qsC' : r'Collapse particle flux, $\displaystyle{q_{sC}} (kg/s)$',
             'clps_regime': 'Collapse Regime',
             'Ze_over_Rv' : r'Scaled water depth, $\displaystyle\frac{Z_e}{r_v}$',
             'hm_over_Q14': r'Scaled plume height, $\displaystyle\frac{h_m}{Q^{1/4}}$',
             'rC_over_Rv' : r'Scaled collapse radius, $\displaystyle\frac{r_C}{r_v}$',
             'qw0_over_Q' : r'Scaled surface water flux, $\displaystyle\frac{q_{w0}}{Q}$',
             'qwC_over_Q' : r'Scaled collapse water flux, $\displaystyle\frac{q_{wC}}{Q}$',
             'qs0_over_Q' : r'Scaled surface particle flux, $\displaystyle\frac{q_{s0}}{Q}$',
             'qsC_over_Q' : r'Scaled collapse particle flux, $\displaystyle\frac{q_{sC}}{Q}$',
             }

scaled_vars = {'Ze': 'Ze_over_Rv',
               'Q': 'logQ',
               'hm' : 'hm_over_Q14',
               'rC' : 'rC_over_Rv',
               'qw0' : 'qw0_over_Q',
               'qwC' : 'qwC_over_Q',
               'qs0' : 'qs0_over_Q',
               'qsC' : 'qsC_over_Q',              
               }

def nrmse(y_true: pd.DataFrame,y_pred: pd.DataFrame) \
        -> Tuple[pd.Series,pd.Series,pd.DataFrame]:
    # Root mean square of error after normalization to the standard deviation
    N_samps = y_true.shape[0]
    
    # Normalized relative error
    norm_error = (y_pred - y_true) # / (y_true - y_pred).std(axis=0)
    
    rmse = np.sqrt((1/N_samps)*(norm_error**2).sum(axis=0))
    
    nrmse = rmse / np.ptp(y_true,axis=0)
    
    return rmse, nrmse, norm_error

def cost_function():
    pass
    
    
def nmse_deprecated(y_true: pd.DataFrame,y_pred: pd.DataFrame) -> Tuple[pd.Series,pd.DataFrame]:
    """
    DEPRECATED
    nmse(y_true,y_pred)
    Normalized mean squared error. 

    Parameters
    ----------
    y_true : pd.DataFrame
        DESCRIPTION.
    y_pred : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    nmse : TYPE
        DESCRIPTION.
    nmse_all : TYPE
        DESCRIPTION.

    """
    zero_th = 0.1 # When close enough to zero is close enough, for this data set
    
    # pred_is_zero = ((y_pred==0) & (y_true!=0))
    true_is_zero = ((y_pred>=zero_th) & (y_true<=zero_th))
    both_are_zero = ((y_pred<=zero_th) & (y_true<=zero_th))
    
    nmse_all = ((y_true - y_pred) / y_true) ** 2
    nmse_all[true_is_zero] = 1.0 #((y_true[true_is_zero] - y_pred[true_is_zero]) / y_pred[true_is_zero]) ** 2
    nmse_all[both_are_zero] = 0
    
    nmse = nmse_all.mean(axis=0, skipna=True)
    
    return nmse, nmse_all


def apply_physical_data_scaling(data: pd.DataFrame, Q_vec: pd.Series = None) \
                                -> pd.DataFrame: 
    '''
    Apply non-demensionalization or scaling to each variable.

    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.
    Q_vec : pd.Series, optional
        Mass flux (Q) vector. Supply when not included in 'data'. Required for 
        non-dimensionalizing.

    Returns
    -------
    out_data : pd.DataFrame
        DataFrame of same size as input data, with corresponding scaled/
        non-dimensionalized values.

    '''
    out_data = data.copy()
    
    if Q_vec is None and type(data!=pd.Series) and data.name!='Q':
        Q_vec = data['Q']  # Must have this if not provided
    
    cols = data.columns
    for col in cols:
        if col in ['Ze','rC']:
            out_data[col] = data[col] / extrapVentRadius(Q_vec)
            
        elif col == 'Q':
           out_data[col] = np.log10(data[col])
            
        elif col == 'hm':
            out_data[col] = data[col] / (Q_vec**(1/4))
            
        elif col in ['qw0','qwC','qs0','qsC']:
            out_data[col] = data[col] / Q_vec

    out_data = out_data.rename(columns=scaled_vars)
                
    return out_data

# Define search functions for querying the test training/test data along an 
# arbitrary coordinate path.
def scale_to_range(input_data: np.ndarray, in_range: tuple, 
                   out_range: tuple = (0,1)) -> np.ndarray:
    """
    Scale a set of data using fixed input and output data ranges.

    Parameters:
    input_data: numpy.ndarray
        Array of scaled 2D data points of shape (N, 2).
    in_range: Array-like
        Tuple of length two specifying range of input data
    out_range Array-lie
        Tuple of length two specifying limits of output data to scale to

    Returns:
    scaled_data: numpy.ndarray
        Array of scaled 2D data points of shape (N, 2).
    """
    scaled_data = (input_data - np.min(in_range)) / \
        (np.max(in_range) - np.min(in_range)) * \
        (np.max(out_range) - np.min(out_range)) + np.min(out_range)
    
    return scaled_data

def get_scaled_coordinates(coord_points : pd.DataFrame, 
                         scale_ranges: dict = None) -> pd.DataFrame:
    """
    Wrapper function for quickly scaling data coordinates

    Parameters
    ----------
    coord_points : pandas.DataFrame
        X, Y, etc data points.
    scale_ranges : dict, optional
        Optional min/max ranges to scale data to. Scales to min/max of the data
        by default.

    Returns
    -------
    scaled_coords : pandas.DataFrame
        Data points scaled to the specified min/max range.

    """
    
    # assert all(scale_ranges.keys() in coord_points.columns)
    
    scaled_coords = coord_points[scale_ranges.keys()]
    if scale_ranges is None:
        scaled_coords = minmax_scale(coord_points, axis=0)
    else:
        for coord in coord_points.columns:
            scaled_coords[coord] = scale_to_range(scaled_coords[coord], scale_ranges[coord] )

    return scaled_coords
    
def average_points_within_radius(coordinate_points, data_points, query_points, 
                                 vector_coords=None, radius=0.05, 
                                 return_all=False) -> \
                                list[pd.DataFrame, pd.DataFrame]:
    """
    Finds all points within a given radius of any query point on the line.

    Parameters:
    -----------
    coordinate_points: numpy.ndarray
        Array of scaled 2D points of shape (M, 2), specifying data coordinates.
    data_points:  numpy.ndarray
        Array of unscaled 2D points of shape (M, N) to retrieve in query
        (N = number of features)
    query_points: numpy.ndarray
        Array of scaled 2D query points along the line of shape (P, 2).
    vector_coords: numpy.ndarray
        Optional of points defining a length vector along the query path. 
        Assigning 'True' will use the first column of coordinate_points
    radius: float
        The normalized radius to check within.

    Returns:
    --------
    average_values: pandas.DataFrame
        (P,N+1) Average of all points within query radius for each query point
    all_values: pandas.DataFrame
        (?,N+1) List of all points in radius query, labeled by query point
    """
    if vector_coords is True:
        vector_coords = query_points[:,0]
    
    # Initialize the nearest neighbors model
    nbrs = NearestNeighbors(radius=radius).fit(coordinate_points)
    
    # Find all points within the radius for each query point
    indices = nbrs.radius_neighbors(query_points, return_distance=False)

    # Get AVERAGE values of test data points with the neighbourhood of each query point
    if vector_coords is not None:
        average_values = pd.concat([data_points.iloc[ind].mean(axis=0) 
                                    for ind in indices],axis=1) \
                                    .transpose().assign(x=vector_coords )
                                    
        # GET ALL values of test data points with the neighbourhood of each query point
        if return_all:
            all_values = pd.concat([data_points.iloc[ind].assign(
                x=np.full( (len(ind),1), ti) )
                for ti,ind in zip(vector_coords,indices)],axis=0)
        else:
            all_values = None                            
                                    
    else:
        average_values = pd.concat(
            [data_points.iloc[ind].mean(axis=0) for ind in indices],axis=1
            ).transpose()
        
        if return_all:
            all_values = pd.concat([data_points.iloc[ind]
                for ti,ind in indices],axis=0)
        else:
            all_values = None
        
    
    
 
    return average_values, all_values




def get_averaged_point_cloud(data_points : pd.DataFrame, 
                             coordinate_columns: list[str],
                             scale_ranges: dict, 
                             radius: float = 0.05) -> pd.DataFrame:
    """
    get_averaged_point_cloud
    Take a point cloud DataFrame, and for all points returns the average from within a neighbourhood radius of the point.

    Parameters
    ----------
    data_points : pandas.DataFrame
        Point cloud data values, including coordinate vectors.
    coordinate_columns : list[str]
        List containing the names of columns to use as point coordinates. Will be used for neighbourhood radius query.
    scale_ranges : dict(tuples)
        Dict with N keys, where N = len(coordinate_columns). Scaling ranges for input data coordinates. Defaults to min_max scaling.
    radius : float
        Normalized search radius for averaging points (relative to scaled 
        coordinate vectors).

    Returns
    -------
    averaged_data_points : pandas.DataFrame
        Data set matching input data_points, with all columns except coordinate
        points averaged.

    """
    # First scale input data points
    scaled_coords = get_scaled_coordinates(data_points[coordinate_columns], 
                                           scale_ranges)
    # scaled_coords = data_points(coordinate_columns)
    # for coord in coordinate_columns:
    #     scaled_coords[coord] = scale_to_range(scaled_coords[coord], scale_ranges[coord] )
    
    # averaged_data_points = None
    
    # data_cols = set(data_points.columns)
    # data_cols = list(data_cols.remove(coordinate_columns))
    data_points = data_points.loc[:, ~data_points.columns.isin(coordinate_columns)]
    
    averaged_data_points,_ = average_points_within_radius(
                                scaled_coords, 
                                data_points, 
                                scaled_coords, 
                                vector_coords=None, 
                                radius=radius, 
                                return_all=False)
    
    return averaged_data_points

def monte_carlo_loss_function(test_data, rf_data, radius=0.05):
    # return loss
    pass

def get_regressor_prediction(regressor,x_predict):
    # return y_predict
    pass


def simulate_model_scenario(
        regressor: RandomForestRegressor,
        test_data: pd.DataFrame,
        Q: tuple                = np.log10((6e6, 4e7)), 
        Ze_max: tuple           = (50, 150),
        coord_ranges            = None,
        input_vars : list[str]  = ['Ze','logQ'],
        output_vars: list[str]  = ['hm','rC','qw0','qwC','qs0','qsC'],
        search_radius: float    = 0.05,
        fig_name: str           = None,
        ):
    '''
    

    Parameters
    ----------
    regressor : RandomForestRegressor
        Trained regressor object to test.
    test_data : pd.DataFrame
        Ground truth.
    Q : tuple, optional
        Vector of test Q values. The default is np.log10((6e6, 4e7)).
    Ze_max : tuple, optional
        Vector of test Ze_max values. The default is (50, 150).
    coord_ranges : TYPE, optional
        Scaling ranges for Q and Ze point coordinates. The default is None, for
        which case the ranges of test_data are used.
    input_vars : list[str], optional
        Name of input "coordinate" vars in DataFrame. The default is ['Ze','logQ'].
    output_vars : list[str], optional
        Name of output vars in DataFrame. 
        The default is ['hm','rC','qw0','qwC','qs0','qsC'].
    search_radius : float, optional
        Averaging radius for deriving ground truth data. The default is 0.05.

    Returns
    -------
    nmse : pd.DataFrame 
        [ len(Ze_max)*len(Q) x len(output_vars) ]
        Normalized mean squared error between simulated true data and RF prediction.
    nsse : pd.DataFrame 
        [ len(Ze_max)*len(Q) x len(output_vars) ]
        Normalized std. dev. squared error between simulated true data and RF prediction.
    roughness_difference : pd.DataFrame 
        [ len(Ze_max)*len(Q) x len(output_vars) ]
        Relative mean roughness difference between the simulated ground truth signal
        and the RF predicted signal (negative values mean the prediction is 
        MORE smooth than the ground truth simulated data).
    std_roughness_difference : pd.DataFrame 
        [ len(Ze_max)*len(Q) x len(output_vars) ]
        Std. dev. of relative roughness difference between the simulated ground truth signal
        and the RF predicted signal (negative values mean the prediction is 
        MORE smooth than the ground truth simulated data).

    '''
    def normalize(y, ref_y=None):
        '''
        Normalize by removing mean and dividing by std. dev.
        Optionally using a reference array from which to obtain the mean and 
        std. dev. values.

        Parameters
        ----------
        y : pd.DataFrame or pd.Series
            Values to normalize.
        ref_y : pd.DataFrame or pd.Series, optional, matches size of y
            Values to normalize by.

        Returns
        -------
        norm_y : pd.DataFrame or pd.Series
            Normalized array.
        '''
        if ref_y is not None:
            assert y.shape==ref_y.shape, 'Normalization reference y does not match size of y.'
            norm_y = ( y - ref_y.mean() ) / ref_y.std()
        else:
            norm_y = ( y - y.mean() ) / y.std()
        return norm_y
    
    def get_mse(y_true,y_pred):
        nmse = ( normalize(y_pred, ref_y=y_true).sub(normalize(y_true))**2).mean(axis=0)
        nsse = ( normalize(y_pred, ref_y=y_true).sub(normalize(y_true))**2).std(axis=0)
        return nmse, nsse
        
    def get_roughness(y):
        # if type(y)==pd.DataFrame:
        #     y_norm = normalizer(y)
        #     mean_roughness = (y_norm.diff(axis=0).diff(axis=0)**2).mean(axis=0)
        #     std_roughness  = (y_norm.diff(axis=0).diff(axis=0)**2).std(axis=0)
        # elif type(y)==pd.Series:
        y_norm = normalize(y)
        mean_roughness = (y_norm.diff().diff()**2).mean()
        std_roughness = (y_norm.diff().diff()**2).std()
        # else:
        #     raise TypeError
        return mean_roughness, std_roughness
    
    def get_roughness_difference(y_true,y_pred):
        assert np.shape(y_true)==np.shape(y_pred), 'Array sizes do not match.'
        if type(y_true)==pd.DataFrame and type(y_pred)==pd.DataFrame:
            assert np.all(y_true.columns==y_pred.columns)
            
        roughness_true, std_roughness_true = get_roughness(y_true)
        roughness_pred, std_roughness_pred = get_roughness(y_pred)
        mean_roughness_difference = roughness_pred - roughness_true
        std_roughness_difference = std_roughness_pred - std_roughness_true
        return mean_roughness_difference, std_roughness_difference

    # def get_roughness_difference(y_true,y_pred):
    #     # Compute mean difference in roughness along columns
    #     # Compute roughness for each as the squared differences
    #     normed_y_true = (y_true - y_true.mean(axis=0)) / y_true.std(axis=0)
    #     roughness_true = (normed_y_true.diff()**2).sum(axis=0)
        
    #     normed_y_pred = (y_pred - y_pred.mean(axis=0)) / y_pred.std(axis=0)
    #     roughness_pred = (normed_y_pred.diff()**2).sum(axis=0)
        
    #     normalized_roughness_difference = roughness_pred - roughness_true
        
    #     return normalized_roughness_difference
    
    assert all([var in test_data.columns for var in input_vars]), \
        "Not all input variables were found in the data set"
    assert all([var in test_data.columns for var in output_vars]), \
        "Not all output variables were found in the data set"
    

    
    # Use a simple exponential function to make a proxy water depth evolution
    # over an arbitrary timescale
    nn = 201
    tau = 1.5
    t = np.linspace(0,10,nn) # Build a dummy time vector
    y = t * np.exp(1 - t/tau)
    
    # Make base array of Ze(t)
    ze_label = [f'Ze_{ze}' for ze in Ze_max]
    
    # Get scaled coordinate points
    coordinate_points = get_scaled_coordinates(test_data[input_vars], coord_ranges)


    # -- Make a big tiled array of simulation outputs and plot each variable --
    num_q = len(Q)
    num_ze = len(Ze_max)
    num_vars = 10
    big_array_size = (nn*num_q,num_vars)
    control_vars = ['t','Ze','Ze_max','logQ'] 
    big_array_columns = control_vars + output_vars
    nmse = pd.DataFrame(data=np.zeros((num_q*num_ze, len(output_vars))), columns=output_vars)
    nsse = pd.DataFrame(data=np.zeros((num_q*num_ze, len(output_vars))), columns=output_vars)
    roughness_difference = pd.DataFrame(data=np.zeros((num_q*num_ze, len(output_vars))), columns=output_vars)
    std_roughness_difference = pd.DataFrame(data=np.zeros((num_q*num_ze, len(output_vars))), columns=output_vars)
    

    
    # Figure setup
    n_cols = len(Ze_max)
    n_rows = len(output_vars)+1
    figsize = (6*num_ze+2,2*(len(output_vars)+1))
    sns.set_theme(style="whitegrid")
    fig,axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                            gridspec_kw={'wspace':0.25, 'hspace':0.04})
    plt.rcParams['text.usetex'] = True

    # Loop over test Ze and Q vectors
    for zi,ze in enumerate(Ze_max):
        big_array = np.zeros(big_array_size)
        Ze_vector = ze * y
        query_dict = {}
        
        for qi,q in enumerate(Q):
            # Get prediction inputs
            x_predict = pd.DataFrame(
                data=np.column_stack((t, Ze_vector, np.full_like(y,ze), np.full_like(y,q))),
                columns=control_vars)
            # Get prediction outputs
            # y_predict = get_regressor_prediction(regressor,x_predict[input_vars])
            y_predict = pd.DataFrame(regressor.predict(x_predict[input_vars]), columns=output_vars)
            iloc1, iloc2 = qi*(nn),(qi+1)*nn
            big_array[iloc1:iloc2, :] = np.concatenate((x_predict,y_predict),axis=1)
    
            # Get training/test data query
            query_points = get_scaled_coordinates(
                pd.DataFrame(np.column_stack((Ze_vector, np.full_like(Ze_vector, q))), 
                             columns=['Ze','logQ']),
                coord_ranges
                )
            
            # query_points = np.column_stack( ( scale_to_range(Ze_vector, Ze_range ), # Ze
            #                              scale_to_range(np.full_like(Ze_vector, q), Q_range)) ) # Q

            # Get the point neighbourhood for each query point
            query_avg, query_dict[str(q)] = average_points_within_radius(
                coordinate_points, test_data[output_vars], query_points,
                vector_coords=t, radius=search_radius, return_all=True)
            
            
            # Add Q labels for plotting
            query_dict[str(q)]=query_dict[str(q)].assign(
                logQ=np.full((query_dict[str(q)].shape[0],1), q))
            
            # sns.lineplot(x='x',y=var,data=all_values,errorbar='sd', ax=ax)
            
            # Calculate error for this query - normalizing by mean of 'true' simulation value
            nmse.iloc[zi*num_ze + qi], nsse.iloc[zi*num_ze + qi] = \
                get_mse(query_avg[output_vars], y_predict)

            # Calculate roughness difference (roughness using ground truth as a benchmark)
            roughness_difference.iloc[zi*num_ze + qi], std_roughness_difference.iloc[zi*num_ze + qi] = \
                get_roughness_difference(query_avg[output_vars], y_predict)
            
            # normalized_mean_error.iloc[zi*num_ze + qi] = \
            #     (y_predict - query_avg[output_vars]).abs().mean(axis=0) / \
            #         query_avg[output_vars].mean(axis=0)
            
            
            # mse.iloc[zi*num_ze + qi] = \
            #     (y_predict.sub(query_avg[output_vars])**2).mean(axis=0)
            # sse.iloc[zi*num_ze + qi] = \
            #     (y_predict.sub(query_avg[output_vars])**2).std(axis=0)
            
            
        query_df = pd.concat(query_dict,axis=0)
        simulation_df = pd.DataFrame(data=big_array, columns=big_array_columns)
    
        # First plot the made-up Ze time series
        sns.lineplot(x=t,y=Ze_vector,ax=axes[0,zi])
        axes[0,zi].set_ylabel(var_labels['Ze'])
        axes[0,zi].set_title(f'$\displaystyle Z_{{e,MAX}} = {ze}$ m')
        
        # Then loop over output variable to plot each
        for vi,var in enumerate(output_vars):  
            # First plot the training/test set query
            # for qi,q in enumerate(Q):
            if vi != 0 or zi != 0:
                show_legend = False
            else:
                show_legend = True

            simp=sns.lineplot(data=query_df,
                         x='x',
                         y=var,
                         hue='logQ',
                         errorbar='sd', 
                         palette=sns.color_palette("Greys", n_colors=num_q),
                         ax=axes[vi+1,zi],
                         legend=show_legend)
                         # label=f'Trainig avg., r = {search_radius}')
            rfp=sns.lineplot(data=simulation_df,
                         x='t',
                         y=var,
                         hue = 'logQ',
                         palette=sns.color_palette("Blues", n_colors=num_q),
                         ax=axes[vi+1,zi],
                         legend=show_legend)
            
            if show_legend:
                lh = simp.get_legend_handles_labels()
                lh[1][0:num_q] = ['Data : ' + lab for lab in lh[1][0:num_q]]
                lh[1][num_q:-1] = ['RF : ' + lab for lab in lh[1][num_q::]]
                rfp.legend(lh[0],lh[1], title=var_labels['logQ'],loc='right')
    
    if fig_name is not None:
        _ = fig.suptitle(fig_name)
            
    return nmse,nsse,roughness_difference,std_roughness_difference

# ------- Helper plotting wrapper functions --------


    
def single_var_panels_scatter(data,
                           x_var: str = 'Ze_over_Rv',
                           y_vars: list = ['hm_over_Q14','rC_over_Rv','qw0_over_Q','qwC_over_Q','qs0_over_Q','qsC_over_Q'],
                           hue_var: str = 'clps_regime',
                           size_var: str = 'logQ',
                           title: str = None,
                           xlim: tuple = None,
                           ):
    
    sns.set_theme(style="whitegrid")
    n_rows = 2
    n_cols = ceil(len(y_vars)/n_rows)
    fig,axes = plt.subplots(n_rows,n_cols, figsize=(18,8),gridspec_kw={'wspace':0.3, 'hspace':0.4})
    # cmap = sns.color_palette('Spectral', as_cmap=True)
    cmap = sns.diverging_palette(30, 250, l=65, center="dark", as_cmap=True)
    if title is not None:
        fig.suptitle(title)
    
    for ax,var in zip(axes.flatten(),y_vars):
        if ax==axes[0,n_cols-1]:
            show_legend = True
            # sc=ax.scatter(data[x_var], data[var],
            # c = data[hue_var],
            # s=10, cmap = cmap)
            # if hue_var=='clps_regime':
            #     cbar = fig.colorbar(sc,ax=axes,ticks=[0,1,2])
            #     cbar.ax.set_yticklabels(['Bouyant','Total Collapse','Steam Plume'])
            #     cbar.ax.invert_yaxis()
            # else:
            #     cbar = fig.colorbar(sc,ax=axes)
                
        else:
            show_legend = False  
        sns.scatterplot(x=x_var,y=var,
                    hue=hue_var,
                    size=size_var,
                    sizes=(10, 100),
                    data=data, 
                    alpha=0.5,
                    palette=cmap, 
                    edgecolors=[0, 0, 0],
                    ax=ax, 
                    legend=show_legend) #"ch:r=-.2,d=.3_r") #, edgecolors=[0, 0, 0])
        
        if xlim is not None:
            ax.set_xlim(left=xlim[0], right=xlim[1],)
        ax.set_xlabel(var_labels[x_var])
        ax.set_ylabel(var_labels[var])
        if show_legend:
            # sns.move_legend(axes[0,-1], "upper right")
            sns.move_legend(axes[0,-1], "upper left", bbox_to_anchor=(1, 0.4))
            
    return fig,axes
    
def double_var_panels_scatter(data,
                           x_var: str = 'Ze_over_Rv',
                           y_vars1: list = ['hm_over_Q14','rC_over_Rv','qw0_over_Q','qwC_over_Q','qs0_over_Q','qsC_over_Q'],
                           y_vars2: list = ['hm_over_Q14_pred','rC_over_Rv_pred','qw0_over_Q_pred','qwC_over_Q_pred','qs0_over_Q_pred','qsC_over_Q_pred'],
                           hue_var: str = 'variable',
                           title: str = None,
                           xlim: tuple = None,
                           ):
    assert len(y_vars1)==len(y_vars2), 'y_var input lists must be of equal length'
    n_rows = 2
    n_cols = ceil(len(y_vars1)/n_rows)
    
    fig,axes = plt.subplots(n_rows, n_cols, figsize=(18,8),gridspec_kw={'wspace':0.3, 'hspace':0.3})
    if title is not None:
        fig.suptitle(title)    
    
    for ax,var1,var2 in zip(axes.flatten(),y_vars1,y_vars2):
        if ax==axes[0,n_cols-1]:
            show_legend = True
        else:
            show_legend = False  
       
        # Plot physical relationships to see what's happening
        sns.scatterplot(
                    data= pd.melt( data[[x_var,var1, var2]], id_vars=x_var, value_vars= [var1, var2]) , 
                     x=x_var,
                    y='value',
                    hue='variable',
                    alpha=0.3,
                    ax=ax, 
                    legend=show_legend,
                    s=10
                    ) 
        
        if xlim is not None:
            ax.set_xlim(left=xlim[0], right=xlim[1],)
        # ax.set_ylabel(var)
        ax.set_xlabel(var_labels[x_var])
        ax.set_ylabel(var_labels[var1])
        if show_legend:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.7))
        
    return fig,axes


def plot_RF_scores(results : pd.DataFrame, savefigs=False, descriptor=None, fig_dir='figures'):
    # PLOTS:
        # Training score w/ std
        # Validation (test) score w/ std
        # Test score (mse)
        # Simulation score (mean error)
        # Roughness score
        # Score time
    def print_fig(fname):
       if descriptor is not None:
           fname = join(fig_dir,f'{descriptor}_{fname}')
       else:
           fname = join(fig_dir,fname)
       plt.savefig(fname, format='pdf')        
    
    plot_fields = ['CV Scores',
              # 'Test MSE',
              'Sim. NMSE',
              'Mean roughness diff.',
              'Mean score time',
              ]
    
    x = results['rank_test_score']
    n_rows = len(plot_fields)
    n_cols = 1

    # Plot scores vs rank
    fig,axes = plt.subplots(n_rows,n_cols, figsize=(12,12)) #,gridspec_kw={'wspace':0.3, 'hspace':0.4})
    for pi,pf in enumerate(plot_fields):
        h  = []
        if pf == 'CV Scores':
            h.append(axes[pi].errorbar(x, results['mean_train_score'],yerr=results['std_train_score'], fmt='o', markersize=4, label='Train'))
            h.append(axes[pi].errorbar(x, results['mean_test_score'],yerr=results['std_test_score'], fmt='o', markersize=4,label='Validation'))
            h.append(axes[pi].plot(x, -results['test_mse'], 'o', markersize=4,label='Test'))
            axes[pi].legend(title='Scores')
            axes[pi].set_ylim((-2e12,0.5e12))
        elif pf == 'Sim. NMSE':
            axes[pi].errorbar(x, results['sim_nmse'],yerr=results['sim_nsse'], fmt='o', markersize=4,label='Simulation')
            axes[pi].set_ylim((0.,1.))
        elif pf == 'Mean roughness diff.':
            axes[pi].plot(x, results['sim_mean_rough_diff'],'o', markersize=4,label='Sim. Roughness')
            # axes[pi].errorbar(x, results['sim_mean_rough_diff'],yerr=results['sim_std_roughness_diff'], fmt='o', markersize=4,label='Sim. Roughness')
        elif pf == 'Mean score time':
            axes[pi].plot(x, results['mean_score_time'],'o', markersize=4,label='Mean score time')

        axes[pi].set_ylabel(plot_fields[pi])
    axes[-1].set_xlabel('Cross Validation Rank')
    
    if savefigs:
        print_fig('scores.pdf')
        
    plt.show()
    

 

    # Make pairplot of scores vs parameters
    x_vars = [col for col in results.columns if 'param_' in col]
    y_vars = ['mean_test_score','mean_train_score','mean_score_time','sim_mean_rough_diff']
    sns.pairplot(results.fillna(0.),x_vars=x_vars,y_vars=y_vars, hue='rank_test_score')
    
    if savefigs:
        print_fig('param_pairplot.pdf')
