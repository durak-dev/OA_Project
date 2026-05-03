import matplotlib.pyplot as plt
import pandas as pd
from algorithms import *
from algorithm_functions import *
from FFO_MLP import OptimizationMLP_Classifier


def compare_algorithms(X, y, configs, n_repeats=30):
    """
    configs: list of dictionaries:
             {'class': AlgoClass, 'params': dict, 'name': str}
    """
    plt.figure(figsize=(10, 6))

    for config in configs:
        all_histories = []
        for _ in range(n_repeats):
            mlp = OptimizationMLP_Classifier(config['class'], hidden_layer_size=10, **config['params'])
            mlp.fit(X, y)
            all_histories.append(mlp.optimizer.history)

        # Convert to numpy array for mean/std calculation
        hist_array = np.array(all_histories)
        mean_hist = np.mean(hist_array, axis=0)
        std_hist = np.std(hist_array, axis=0)

        # Plotting
        iters = range(len(mean_hist))
        plt.plot(iters, mean_hist, label=config['name'])
        plt.fill_between(iters, mean_hist - std_hist, mean_hist + std_hist, alpha=0.2)

    plt.title('Algorithm Performance Comparison (Mean +/- Std)')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Loss)')
    plt.legend()
    plt.show()
    plt.savefig(f'figures/algorithm_performance{configs}.png')

if __name__ == "__main__":
    URL = r"C:\Users\Lucas\Documents\NOVA IMS\2nd Year\Machine Learning\Final Project\ML15-All_Files\Nata_Files\less_features"
    nata_train = pd.read_csv(URL + r'\train_data_lf.csv', index_col=0)
    nata_test = pd.read_csv(URL + r'\test_data_lf.csv', index_col=0)
    # X = park_data.iloc[:, :-1]
    # y = park_data.iloc[:, -1]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_test = nata_test.drop(columns='target')
    y_test = nata_test.target.map({'OK': 1, 'KO': 0})

    X_train = nata_train.drop(columns='target')
    y_train = nata_train.target.map({'OK': 1, 'KO': 0})


    ga_params = {'selection': tournament_selection, 'crossover': alpha_crossover,
                 'mutation': adaptive_mutation, 'pop_size': 50,
                 'n_iter': 1000, 'mutation_rate': 0.1, 'maximize': False}

    de_params = {'mutation': diff_evol_mutation, 'pop_size': 50,
                 'n_iter': 1000, 'mutation_factor': 0.9, 'maximize': False}

    configs = [
        {'class': GeneticAlgorithm, 'params': ga_params, 'name': 'Genetic Algorithm'},
        {'class': DifferentialEvolution, 'params': de_params, 'name': 'Differential Evolution'}
    ]

    # Run comparison
    compare_algorithms(X_train, y_train, configs)