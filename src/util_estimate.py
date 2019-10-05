import pickle
import os
import pandas as pd

UTIL_FILE = 'util.pkl'
util_matrix = None


def create_util_matrix(args):
    # parse the stop data
    stop_data = pd.read_csv(args.stops_path,
                            usecols=['Est_TotPop_Density','CorrespondingStopID', 'Transfer'])
    #stop_data.dropna(inplace=True)
    stop_data.rename(columns={'CorrespondingStopID': 'StopId'}, inplace=True)
    
    # parse the demographic data
    demo_data = pd.read_csv(args.demo_path,
                            usecols=['StopId','Routes', 'Est_Pop_Density_SqMile'])

    # parse the ridership data
    ridership_data = pd.read_csv(args.rider_path,
                                 usecols=['StopID','IndividUtilization','IndividRoute','NumberOfRoutes'])
    ridership_data.rename(columns={'StopID': 'StopId'}, inplace=True)
    ridership_data['IndividUtilization'] = ridership_data['IndividUtilization'].astype(float)
    
    

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
