import pandas as pd
from sklearn.model_selection import train_test_split
from algorithm_functions import *
from algorithms import GeneticAlgorithm, DifferentialEvolution, ParticleSwarm
from pararllel_testing import compare_algorithms
# from testing import compare_algorithms     # Switch to parallel if something goes wrong with the non-parallel version
from FFO_MLP import OptimizationMLP_Classifier


park_data = pd.read_csv('parkinsons_preprocessed.csv', index_col=0)

X = park_data.iloc[:, :-1]
y = park_data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

ga_params ={'selection': tournament_selection,
            'crossover': alpha_crossover,
            'mutation': adaptive_mutation,
            'pop_size': 100,
            'n_iter': 1000,
            'mutation_rate': 0.1,
            'maximize': False}

de_params ={'mutation': diff_evol_mutation,
            'crossover': binomial_crossover,
            'pop_size': 100,
            'n_iter': 1000,
            'mutation_factor': 0.8,
            'crossover_rate': 0.9,
            'maximize': False}

pso_params = {'pop_size': 40,        
            'n_iter': 250,         
            'w': 0.6,              
            'c1': 1.4,             
            'c2': 1.4,             
            'v_max': 0.05,         
            'w_limit': 1.5,        
            'maximize': False}

configs = [
        {'class': GeneticAlgorithm, 'params': ga_params, 'name': 'Genetic Algorithm'},
        {'class': DifferentialEvolution, 'params': de_params, 'name': 'Differential Evolution'},
        {'class': ParticleSwarm, 'params': pso_params, 'name': 'Particle Swarm'}
    ]

# Run comparison


"""GA_opt = OptimizationMLP_Classifier(GeneticAlgorithm, X_train.shape[1], error_calc="BCE" ,**ga_params)
GA_opt.fit(X_train, y_train)
print(GA_opt.score(X_train, y_train))
print(GA_opt.score(X_test, y_test))
GA_opt.history()
DE_opt = OptimizationMLP_Classifier(DifferentialEvolution, X_train.shape[1], weight_init="glorot" , **de_params)
DE_opt.fit(X_train, y_train)
print(DE_opt.score(X_train, y_train))
print(DE_opt.score(X_test, y_test))
DE_opt.history()"""
compare_algorithms(X_train, y_train, X_test, y_test, configs, n_repeats=30)

