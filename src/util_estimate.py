import pickle
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

UTIL_FILE = 'util.pkl'
util_matrix = None


def create_util_matrix(args, init_model=LinearRegression, preprocessing=None):
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

    model = init_model()
    model.fit(indeps, dep)
    print('Model trained with R^2 = {}'.format(model.score(indeps, dep)))

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
        util_matrix = create_util_matrix(args, preprocessing=np.log)
        print('Utilization matrix created')


def get_utilization(route):
    if util_matrix is None:
        raise(ValueError('You have not yet loaded the utilization matrix'))
    else:
        # Eventually replace with code to return utilization
        pass
