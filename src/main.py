#!/usr/bin/env python3

import argparse
import pandas as pd
import glob
import os

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

def main():
    args = parse_args()
    
    data = parse_data(args.input)
    print(data)

if __name__ == '__main__':
    main()
