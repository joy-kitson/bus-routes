import argparse
import random
import pickle

from deap import base, creator, tools

import util_estimate
import time_estimate

random.seed(42)
CURRENT_ROUTE = []  # fill this in with real current route
NUM_STOPS = len(CURRENT_ROUTE)


def parse_args():
    '''
    Defines the parser for the command line arguments of the file

    Returns:
        Namespace object containing command line arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('util_weight', type=float, nargs=1,
                        help='The weight of the utilization in the route score')

    parser.add_argument('-ps', '--pop_size',
                        type=int, nargs=1, default=500,
                        help='The population size for the genetic algorithm')

    parser.add_argument('-t', '--num_iterations',
                        type=int, nargs=1, default=1000,
                        help='The population size for the genetic algorithm')

    parser.add_argument('-mp', '--sol_mut_prob',
                        type=float, nargs=1, default=0.2,
                        help='The mutation probability for entire solutions for the genetic algorithm')

    parser.add_argument('-indpb', '--ind_mut_prob',
                        type=float, nargs=1, default=0.05,
                        help='The mutation probability for individual genes in a solution for the genetic algorithm')

    parser.add_argument('-cp', '--crossover_prob',
                        type=float, nargs=1, default=0.5,
                        help='The crossover probability for the genetic algorithm')

    parser.add_argument('-ts', '--tournament_size',
                        type=int, nargs=1, default=3,
                        help='The tournament size for the genetic algorithm')

    return parser.parse_args()


def route_score(candidade_route, original_util, original_time):
    return (util_estimate.get_utilization(candidade_route) / original_util,
            original_time / time_estimate.get_time(candidade_route))


def main():
    args = parse_args()

    time_estimate.load()
    util_estimate.load()

    original_util = util_estimate.get_utilization(CURRENT_ROUTE)
    original_time = time_estimate.get_time(CURRENT_ROUTE)

    # Register fitness measure and individual type with creator
    creator.create("Route_Fitness", base.Fitness,
                   weights=(args.util_weight, 1.0 - args.util_weight))
    creator.create("Individual", list, fitness=creator.Route_Fitness)

    toolbox = base.Toolbox()

    # Add attribute, individual, and population types to toolbox
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, NUM_STOPS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Add evaluation, crossover, mutation, and selection to toolbox
    toolbox.register("evaluate", route_score, original_util, util_matrix,
                     original_time, time_matrix)
    toolbox.register("crossover", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, args.ind_mut_prob)
    toolbox.register("select", tools.selTournament, args.tournament_size)


if __name__ == '__main__':
    main()
