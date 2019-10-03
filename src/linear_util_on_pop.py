#!/usr/bin/env python3

import numpy as np
import math
import argparse
import glob
import os
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

ROUTE_NUM = 25
ROUTE_FILE = 'route25_potential_and_real_stops'
RIDERSHIP_FILE = 'Stop_Riders_Ranking_by_Route_Daily_Totals_May_2019'


def parse_args():
    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument('-i', '--input', nargs='+',
                        default=glob.glob('../data/*.csv'),
                        help='a list of csv files to parse for input data')

    return parser.parse_args()


def parse_data(files):
    # load csv files and parse them in pandas dataframes
    data = {os.path.basename(f).split('.')[0]: pd.read_csv(f) for f in files}

    return data


def is_int(n):
    try:
        int(n)
        return True
    except:
        return False


def train_model(data, init_model=LinearRegression):
    # extract the stop data for the current route
    route = data[ROUTE_FILE]
    cur_route = route.loc[route['CorrespondingStopID'].notnull()]
    cur_route.loc[:, 'CorrespondingStopID'] = cur_route['CorrespondingStopID'].apply(int) 
    # cur_route['Transfer'] = cur_route['Transfer'].astype(bool)
    print(cur_route)

    # extract the ridership data for the current route
    ridership = data[RIDERSHIP_FILE]
    cur_ridership = ridership.loc[ridership['IndividRoute'] == ROUTE_NUM]
    cur_ridership['TOTAL'] = pd.to_numeric(cur_ridership['TOTAL'])
    print(cur_ridership)

    # we need a single column name to merge on, so make sure we call the stop
    # id the same thing in both dataframes
    cur_ridership['CorrespondingStopID'] = cur_ridership['UNIQUE_STOP_NUMBER']
    cur_route_ridership = cur_ridership.merge(route, on='CorrespondingStopID')
    print(cur_route_ridership)

    # select our dependent and independent variables
    i_cols = [
        'Est_TotPop',
        # 'Est_TotMinority',
        # 'Est_TotPov',
        # 'Est_TotLEP',
        # 'Est_TotPop_Density',
        # 'Transfer'
    ]
    indeps = cur_route_ridership[i_cols] 
    dep = cur_route_ridership['TOTAL']

    # train the model
    reg = init_model()
    reg.fit(indeps, dep)

    cur_route_ridership['Predicted'] = reg.predict(indeps)
    ax = cur_route_ridership.plot(x='Est_TotPop', y='Predicted', kind='line', color='r')
    cur_route_ridership.plot(x='Est_TotPop', y='TOTAL', kind='scatter', ax=ax)

    print(reg.score(indeps, dep))
    print(cur_route_ridership.dtypes)

    plt.savefig('../plots/fit.png') 


def main():
    args = parse_args()

    data = parse_data(args.input)

    train_model(data)
    # train_model(data, init_model=lambda : make_pipeline(PolynomialFeatures(3), Ridge()))


if __name__ == '__main__':
    main()
