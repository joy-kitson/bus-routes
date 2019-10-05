import pickle
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

UTIL_FILE = 'util.pkl'
util_matrix = None


def create_util_matrix(args, init_model=LinearRegression):
    # parse the stop data
    stop_data = pd.read_csv(args.stops_path,
                            usecols=['Est_TotPop_Density','CorrespondingStopID', 'Transfer'])
    #stop_data.dropna(inplace=True)
    stop_data.rename(columns={'CorrespondingStopID': 'StopId',
                              'Est_TotPop_Density': 'Est_Pop_Density_SqMile'},
                     inplace=True)

    # parse the demographic data
    demo_data = pd.read_csv(args.demo_path,
                            usecols=['StopId','Routes', 'Est_Pop_Density_SqMile'])
    demo_data['Est_Pop_Density_SqMile'] = demo_data['Est_Pop_Density_SqMile'].astype(float)

    # parse the ridership data
    ridership_data = pd.read_csv(args.rider_path,
                                 usecols=['StopID','IndividUtilization','IndividRoute','NumberOfRoutes'])
    ridership_data.rename(columns={'StopID': 'StopId'}, inplace=True)
    ridership_data['IndividUtilization'] = ridership_data['IndividUtilization'].astype(float)
    
    # set up the data so we can build our model
    training_data = pd.merge(demo_data, ridership_data, on='StopId')
    training_data.dropna(inplace=True)
    indep_cols = ['Est_Pop_Density_SqMile']
    indeps = training_data[indep_cols]
    dep = training_data['IndividUtilization']

    model = init_model()
    model.fit(indeps, dep)
    print('Model trained with R^2 = {}'.format(model.score(indeps, dep)))

    utils = model.predict(stop_data[indep_cols])
    # TODO: finish getting actual ridership data for existing stops
    for i, row in stop_data.iterrows():
        if not np.isnan(row['CorrespondingStopID']):
            utils[i] 

    return utils

def load(args):
    try:
        with open(os.path.join(args.cache_path, UTIL_FILE)) as f:
            util_matrix = pickle.load(f)
    except:
        print("Utilization matrix not found, creating one now")
        util_matrix = create_util_matrix(args)
        print('Utilization matrix created')


def get_utilization(route):
    if util_matrix is None:
        raise(ValueError('You have not yet loaded the utilization matrix'))
    else:
        # Eventually replace with code to return utilization
        pass
