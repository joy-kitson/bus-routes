#!/usr/bin/env python3

import argparse
import glob
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

ROUTE_NUM = 25
ROUTE_FILE = 'route25_potential_and_real_stops'
RIDERSHIP_FILE = 'Stop_Riders_Ranking_by_Route_Daily_Totals_May_2019'

def parse_args():
    parser = argparse.ArgumentParser()

    #optional arguments
    parser.add_argument('-i', '--input', nargs='+', default=glob.glob('../data/*.csv'), \
                        help='a list of csv files to parse for input data')

    return parser.parse_args()

def parse_data(files):
    #load csv files and parse them in pandas dataframes
    data = {os.path.basename(f).split('.')[0]: pd.read_csv(f) for f in files}

    return data

def is_int(n):
    try:
        int(n)
        return True
    except:
        return False

def train_lin_reg(data):
    # extract the stop data for the current route
    route = data[ROUTE_FILE]
    cur_route = route.loc[route['CorrespondingStopID'].notnull()]
    print(cur_route)

    ridership = data[RIDERSHIP_FILE]
    cur_ridership = ridership.loc[ridership['IndividRoute'] == ROUTE_NUM]
    print(cur_ridership)

    reg = LinearRegression() 

def main():
    args = parse_args()
    
    data = parse_data(args.input)
    
    train_lin_reg(data)


if __name__ == '__main__':
    main()
