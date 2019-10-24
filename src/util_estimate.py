import pickle
import os
import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

UTIL_FILE = 'util.pkl'
util_matrix = None


def create_util_matrix(args, model=LinearRegression(n_jobs=-1), preprocessing=None):
    # parse the stop data
    stop_data = pd.read_csv(args.stops_path,
                            usecols=['Est_TotPop_Density','Transfer', 'Id2'])
    #stop_data.dropna(inplace=True)
    stop_data.rename(columns={'Est_TotPop_Density': 'Est_Pop_Density_SqMile',
                              'Id2': 'Formatted FIPS'},
                     inplace=True)

    # parse the demographic data
    demo_data = pd.read_csv(args.demo_path,
                            usecols=['StopId','Routes', 'Est_Pop_Density_SqMile'])
    demo_data['Est_Pop_Density_SqMile'] = demo_data['Est_Pop_Density_SqMile'].astype(float)

    #parse the employment data
    emp_data = pd.read_csv(args.emp_path,
                           usecols=['StopId', 'NumberWorkers', 'Formatted FIPS',
                                    'Est_TotPov_PostRemovedDupIntersects',
                                    'Est_TotLEP_PostRemovedDupIntersects',
                                    'Est_TotMinority_PostRemovedDupIntersects',
                                    ]
    )

    # parse the ridership data
    ridership_data = pd.read_csv(args.rider_path,
                                 usecols=['StopID','IndividUtilization','IndividRoute','NumberOfRoutes'])
    ridership_data.rename(columns={'StopID': 'StopId'}, inplace=True)
    ridership_data['IndividUtilization'] = ridership_data['IndividUtilization'].astype(float)

    # set up the data so we can build our model
    training_data = pd.merge(demo_data, emp_data, on='StopId')
    training_data = pd.merge(training_data, ridership_data, on='StopId')
    indep_cols = [
        'Est_Pop_Density_SqMile',
        'NumberWorkers',
        'Est_TotPov_PostRemovedDupIntersects',
        'Est_TotLEP_PostRemovedDupIntersects',
        'Est_TotMinority_PostRemovedDupIntersects',
    ]
    dep_col = 'IndividUtilization'

    # perform preprocessing
    if preprocessing:        
        training_data[indep_cols + [dep_col]] = \
                training_data[indep_cols + [dep_col]].apply(preprocessing)
        training_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    training_data.dropna(inplace=True)
    
    indeps = training_data[indep_cols]
    dep = training_data['IndividUtilization']

    # perform k-fold cross validation to get our R^2 value
    kf = KFold(n_splits=10)
    abs_err = 0
    for train_indices, test_indices in kf.split(indeps):
        fold_model = deepcopy(model)
        fold_model.fit(indeps.iloc[train_indices], dep.iloc[train_indices])
        fold_preds = fold_model.predict(indeps.iloc[test_indices])
        abs_err += mean_absolute_error(dep.iloc[test_indices], fold_preds)
    n_folds = kf.get_n_splits(indeps)
    abs_err /= n_folds
    print('Model trained with mean absolute error of {} across {} folds'.format(abs_err, n_folds))

    model.fit(indeps, dep)
    print('On full data set, model has an R^2 value of {}'.format(model.score(indeps, dep)))

    #merge in extra demographic data for the stops
    stop_data = pd.merge(stop_data, emp_data, on='Formatted FIPS')
    
    utils = model.predict(stop_data[indep_cols])
    # TODO: finish getting actual ridership data for existing stops
    for i, row in stop_data.iterrows():
        if not np.isnan(row['StopId']):
            utils[i] = ridership_data[ridership_data['StopId'] == row['StopId']]['IndividUtilization'].values[0]

    return utils


def load(args):
    # We assign util_matrix a value within this function, so we need to explicit state that it's global
    global util_matrix
    
    try:
        with open(os.path.join(args.cache_path, UTIL_FILE)) as f:
            util_matrix = pickle.load(f)
    except:
        print("Utilization matrix not found, creating one now")
        #util_matrix = create_util_matrix(args)
        #util_matrix = create_util_matrix(args, model=KNeighborsRegressor(n_neighbors=5, \
        #                                                                 weights='distance', \
        #                                                                 p=2, \
        #                                                                 n_jobs=-1))
        util_matrix = create_util_matrix(args, model=RandomForestRegressor(\
                                                     n_estimators=100, \
                                                     criterion='mae',\
                                                     #min_samples_leaf=3,\
                                                     max_depth=8,\
                                                     n_jobs=-1))
        print('Utilization matrix created')


def get_utilization(route):
    if util_matrix is None:
        raise(ValueError('You have not yet loaded the utilization matrix'))
    else:
        # Eventually replace with code to return utilization
        pass
