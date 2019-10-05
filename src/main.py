#!/usr/bin/env python3

import argparse
import random
import pickle
import os

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

    # Positional args:
    parser.add_argument('util_weight', type=float, nargs=1,
                        help='The weight of the utilization in the route score')

    # Optional args:
     
    # parameters for genetic algorithm
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

    # paths for various files
    parser.add_argument('-d_p', '--demo_path',
                         nargs=1, default=os.path.join('..', 'data', 'demographics.csv'),
                         help='The path to the csv file containing the demographic data')
    parser.add_argument('-r_p', '--rider_path',
                         nargs=1, default=os.path.join('..', 'data', 'ridership.csv'),
                         help='The path to the csv file containing the ridership data')
    parser.add_argument('-s_p', '--stops_path',
                         nargs=1, default=os.path.join('..', 'data', 'stops.csv'),
                         help='The path to the csv file containg all of the potential and current'\
                              + ' stops')
    parser.add_argument('-c_p', '--cache_path', nargs=1, default=os.path.join('..', 'cache'),
                         help='The path to the directory in which to load and store the cached utility'\
                              + ' and travel time matrices')

    return parser.parse_args()


def route_score(candidade_route, original_util, original_time):
    return (util_estimate.get_utilization(candidade_route) / original_util,
            original_time / time_estimate.get_time(candidade_route))


def main():
    args = parse_args()

    time_estimate.load()
    util_estimate.load(args)

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

    print("Start of evolution")
    pop = toolbox.population(n=args.pop_size)

    # Evaluate the entire population
    fitnesses = [toolbox.evaluate(ind) for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extracting all the fitnesses of the individuals
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < args.num_iterations:
        # A new generation
        g = g + 1
        # print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
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

        '''
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        '''

    print("-- End of evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == '__main__':
    main()
