#!/usr/bin/env python3

import argparse
import random
import pickle
import os
from datetime import datetime

import pandas as pd
import numpy as np

from deap import base, creator, tools

import util_estimate
import time_estimate

random.seed(42)


def parse_args():
    '''
    Defines the parser for the command line arguments of the file

    Returns:
        Namespace object containing command line arguments
    '''
    parser = argparse.ArgumentParser()

    # Positional args:
    parser.add_argument('util_weight', type=float, nargs=1,
                        help='The weight of the utilization in the route score')

    # Optional args:

    # parameters for genetic algorithm
    parser.add_argument('-ps', '--pop_size',
                        type=int, nargs=1, default=400,
                        help='The population size for the genetic algorithm')
    parser.add_argument('-t', '--num_iterations',
                        type=int, nargs=1, default=600,
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

    # paths for various files
    parser.add_argument('-s_p', '--stops_path',
                         nargs=1, default=os.path.join('..', 'data',
                                                       'route_25_valid.csv'),
                         help='The path to the csv file containg all of the potential and current'\
                              + ' stops')
    parser.add_argument('-d_p', '--demo_path',
                         nargs=1, default=os.path.join('..', 'data', 'DemoByStops.csv'),
                         help='The path to the csv file containing the demographic data')
    parser.add_argument('-r_p', '--rider_path',
                         nargs=1, default=os.path.join('..', 'data',
                                                       'Stop_Riders_Ranking_by_Route_Daily_Totals_May_2019.csv'),
                         help='The path to the csv file containing the ridership data')
    parser.add_argument('-c_p', '--cache_path', nargs=1, default=os.path.join('..', 'cache'),
                         help='The path to the directory in which to load and store the cached utility'\
                              + ' and travel time matrices')
    parser.add_argument('-l_p', '--log_path', nargs=1, default=os.path.join('..', 'logs'),
                         help='The path to the directory in which to store the results log')

    return parser.parse_args()


def valid_route(potential_route, distances):
    last_stop = 0
    for index, value in enumerate(potential_route):
        if value == 1:
            next_stop = distances[index]
            diff = next_stop - last_stop
            if diff < 500:
                return False
            else:
                last_stop = next_stop
    return True


def log_results(folder_path, mins, maxes, means, st_devs, best_solution):
    now = datetime.now()
    filename = now.strftime('results_%m_%d_%H_%M')
    txt_path = os.path.join(folder_path, filename + '.txt')
    csv_path = os.path.join(folder_path, filename + '.csv')

    with open(txt_path, 'w') as f:
        f.write('Route 25 Optimization\n')
        f.write(now.strftime('Run on %m/%d at %H:%M\n'))
        f.write(f'Best route: {best_solution}\n')
        f.write(f'Best route fitness: {best_solution.fitness.values}\n')

        for index, stats in enumerate(zip(mins, maxes, means, st_devs)):
            (gen_min, gen_max, gen_mean, gen_std) = stats
            f.write(f'------------------- Generation {index} -------------------\n')
            f.write(f'Min fitness: {gen_min}, Max fitness: {gen_max}\n')
            f.write(f'Average Fitness: {gen_mean}, Fitness Standard Deviation: {gen_std}\n')

    results = pd.DataFrame({'Generation': list(range(len(mins))),
                            'Min Score': mins,
                            'Max Score': maxes,
                            'Mean Score': means,
                            'Standard Deviation': st_devs})
    results.to_csv(csv_path, index=False)


def main():
    args = parse_args()

    r25 = pd.read_csv(args.stops_path)

    transfer = r25[r25['Transfer'] == 'Yes']
    non_transfer = r25[r25['Transfer'] == 'No']
    current_route = list(np.logical_not(np.isnan(non_transfer['CorrespondingStopID'].values)).astype(int))
    NUM_STOPS = len(current_route)

    non_transfer_distances = list(r25[r25['Transfer'] == 'No']['Distance from initial stop (feet)'])

    time_estimate.load(args)
    util_estimate.load(args)

    def stop_indices(route):
        indices = [non_transfer.index[i] for (i, val) in enumerate(route) if val == 1]
        indices += list(transfer.index.values)
        indices.sort()
        return indices

    original_util = util_estimate.get_utilization(stop_indices(current_route))
    original_time = time_estimate.get_time(stop_indices(current_route))

    def route_score(candidade_route, util_weight=0.5):
        if valid_route(candidade_route, non_transfer_distances):
            indices = stop_indices(candidade_route)
            return ((util_estimate.get_utilization(indices) / original_util) * util_weight + \
                (original_time / time_estimate.get_time(indices)) * (1 - util_weight),)
        else:
            return (0,)

    # Register fitness measure and individual type with creator
    creator.create("Route_Fitness", base.Fitness,
                   weights=(1.0,))
    creator.create("Individual", list, fitness=creator.Route_Fitness)

    toolbox = base.Toolbox()

    # Add attribute, individual, and population types to toolbox
    toolbox.register("attr_bool", np.random.choice, [0, 1], p=[.95, .05])
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_bool, NUM_STOPS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Add evaluation, crossover, mutation, and selection to toolbox
    toolbox.register("evaluate", route_score,
                     util_weight=args.util_weight[0])
    toolbox.register("crossover", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=args.ind_mut_prob)
    toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)
    toolbox.register("mate", tools.cxTwoPoint)

    print("Start of evolution")
    pop = toolbox.population(n=args.pop_size)

    # Evaluate the entire population
    fitnesses = [toolbox.evaluate(ind) for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Variable keeping track of the number of generations
    g = 0

    maxes, mins, means, st_devs = [], [], [], []

    # Begin the evolution
    while g < args.num_iterations:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < args.crossover_prob:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < args.sol_mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # print("Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        mins.append(min(fits))
        maxes.append(max(fits))
        means.append(mean)
        st_devs.append(std)

    print("-- End of evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    log_results(args.log_path, mins, maxes, means, st_devs, best_ind)


if __name__ == '__main__':
    main()
