import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithm_functions import *
from algorithms import GeneticAlgorithm, DifferentialEvolution, ParticleSwarm
#if you're worried about your computer crashing switch to testing
from testing import compare_algorithms_parallel, plot_tukey_distributions, run_tukey_hsd
from FFO_MLP import OptimizationMLP_Classifier
from report_generation import compile_saved_results_to_pdf


park_data = pd.read_csv('parkinsons_preprocessed.csv', index_col=0)

X = park_data.iloc[:, :-1]
y = park_data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ga_params ={'selection': tournament_selection,
            'crossover': alpha_crossover,
            'mutation': adaptive_mutation,
            'pop_size': 100,
            'n_iter': 1000,
            'mutation_rate': 0.05,
            'maximize': False}

de_params ={'mutation': diff_evol_mutation,
            'crossover': binomial_crossover,
            'pop_size': 100,
            'n_iter': 1000,
            'mutation_factor': 0.6,
            'crossover_rate': 0.8,
            'maximize': False}

pso_params = {'pop_size': 100,
            'n_iter': 1000,
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


"""GA_opt = OptimizationMLP_Classifier(GeneticAlgorithm, X_train_scaled.shape[1], error_calc="BCE" ,**ga_params)
GA_opt.fit(X_train_scaled, y_train)
print('training:', GA_opt.score(X_train_scaled, y_train))
print('testing:', GA_opt.score(X_test_scaled, y_test))
DE_opt = OptimizationMLP_Classifier(DifferentialEvolution, X_train_scaled.shape[1], weight_init="glorot" , **de_params)
DE_opt.fit(X_train_scaled, y_train)
print('training:', DE_opt.score(X_train_scaled, y_train))
print('testing:', DE_opt.score(X_test_scaled, y_test))
PSO = OptimizationMLP_Classifier(ParticleSwarm, X_train_scaled.shape[1], **pso_params)
PSO.fit(X_train_scaled, y_train)
print('training:', PSO.score(X_train_scaled, y_train))
print('testing:', PSO.score(X_test_scaled, y_test))"""


#only run this if you don't need your computer for at least 10 minutes
df_history, df_scores = compare_algorithms_parallel(X_train_scaled, y_train, X_test_scaled, y_test, configs, n_repeats=30)
df_history.to_csv('data/algorithm_performance_history.csv')
df_scores.to_csv('data/algorithm_performance_scores.csv')
test_scores = df_scores[df_scores['Type']=='Testing']
test_scores.to_csv('data/test_scores.csv')
run_tukey_hsd(test_scores)
plot_tukey_distributions(test_scores)
compile_saved_results_to_pdf(
    data_path="data/test_scores.csv",
    figures_dir="figures",
    configs=configs,
    output_pdf="final_experiment_report.pdf"
)