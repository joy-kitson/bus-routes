import pickle
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

UTIL_FILE = 'util.pkl'
util_matrix = None


def create_util_matrix(args, init_model=LinearRegression):
    pass


def load(args):
    pass


def get_utilization(route):
    return len(route) * 4
