import pickle
import os
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from seaborn import scatterplot, distplot
from yellowbrick.regressor import ResidualsPlot

UTIL_FILE = '{}_util.pkl'
util_matrix = np.array([])

# each estimator should be an Scikit-Learn Regressor (or at least follow that API)
UTIL_ESTIMATORS = {
    'linreg': LinearRegression(n_jobs=-1),
    'knn': KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        p=2,
        n_jobs=-1
    ),
    'forest': RandomForestRegressor(\
        n_estimators=100, \
        criterion='mae',\
        min_samples_leaf=.005,\
        #max_depth=8,\
        n_jobs=-1
    ),
}

FULL_NAMES = {
    'linreg': 'Linear Regression',
    'knn': 'K Nearest Neighbors',
    'forest': 'Random Forest'
}

def create_util_matrix(args, model, preprocessing=None):
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
    fold_actuals = []
    fold_ests = []
    fold_resids = []
    for train_indices, test_indices in kf.split(indeps):
        fold_model = deepcopy(model)
        fold_model.fit(indeps.iloc[train_indices], dep.iloc[train_indices])
        fold_preds = fold_model.predict(indeps.iloc[test_indices])
        abs_err += mean_absolute_error(dep.iloc[test_indices], fold_preds)

        if args.plot_util:
            # gather test results for plotting
            fold_actuals.append(dep.iloc[test_indices])
            fold_ests.append(fold_preds)
            fold_resids.append(fold_preds - dep.iloc[test_indices])
    n_folds = kf.get_n_splits(indeps)
    abs_err /= n_folds
    print('Model trained with mean absolute error of {} across {} folds'.format(abs_err, n_folds))

    if args.plot_util:
        fold_actuals = np.concatenate(fold_actuals)
        fold_ests = np.concatenate(fold_ests)
        fold_resids = np.concatenate(fold_resids)

        ax = scatterplot(x=fold_actuals, y=fold_resids)
        ax.set(xlabel='True Average Utilization',
               ylabel='Predicted Average Utilization',
               title='Residuals for {} Model'.format(FULL_NAMES[args.util_estimator]))
        plt.savefig('{}_resids_plot.png'.format(args.util_estimator))

        plt.hist(fold_ests)
        ax.set(xlabel='Predicted Average Utilization',
               ylabel='Frequency',
               title='Distribution of {} Utilization Predictions'.format(FULL_NAMES[args.util_estimator]))
        plt.savefig('{}_histogram.png'.format(args.util_estimator))


    model.fit(indeps, dep)
    print('On full data set, model has an R^2 value of {}'.format(model.score(indeps, dep)))

    #merge in extra demographic data for the stops
    stop_data = pd.merge(stop_data, emp_data, on='Formatted FIPS')
    
    utils = model.predict(stop_data[indep_cols])
    
    # For now, use the predictions instead of the actual utilization, since our predictions tend to be larger than
    # the actual values
    # for i, row in stop_data.iterrows():
    #    if not np.isnan(row['StopId']):
    #        utils[i] = ridership_data[ridership_data['StopId'] == row['StopId']]['IndividUtilization'].values[0]

    return utils


def load(args):
    # We assign util_matrix a value within this function, so we need to explicit state that it's global
    global util_matrix
    
    pickle_path = os.path.join(args.cache_path, UTIL_FILE.format(args.util_estimator))
    
    if not args.force_util:
        try:
            with open(pickle_path, 'rb') as f:
                print('Attempting to load utilization matrix')
                util_matrix = pickle.load(f)
            print('Utilization matrix loaded successfully')
        except:
            print("Utilization matrix could not be loaded")
    
    if not util_matrix.any():
        print('Attempting to create new utilization matrix')
        
        util_matrix = create_util_matrix(args, UTIL_ESTIMATORS[args.util_estimator])
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(util_matrix, f)
        
        print('Utilization matrix created')


def get_utilization(route):
    return sum(util_matrix[route])


def get_individual_util(route):
    return util_matrix[route]
