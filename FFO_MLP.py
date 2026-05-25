import numpy as np
import pandas as pd
from algorithm_functions import *


class OptimizationMLP_Classifier:
    def __init__(self, algorithm_class, hidden_layer_size, weight_init='gaussian', error_calc='BCE', random_state=None,
                 **algo_params):
        """
        algorithm_class: The UNINSTANTIATED class (e.g., GeneticAlgorithm)
        algo_params: A dictionary of hyperparameters (e.g., {'pop_size': 50})
        hidden_layer_size: The number of neurons in the hidden layer
        """
        self.algorithm_class = algorithm_class
        self.hidden_layer_size = hidden_layer_size
        self.weight_init = weight_init
        self.error_calc = error_calc
        self.model_weights = None
        self.rng = np.random.default_rng(random_state)
        self.algo_params = algo_params
        self.algo_params['random_state'] = self.rng

    def _forward(self, X, weights):
        """Internal forward pass logic."""
        W1, b1, W2, b2 = weights
        z1 = np.dot(X, W1) + b1
        a1 = np.tanh(z1)
        return np.dot(a1, W2) + b2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _fitness_wrapper(self, weights, X, y):
        """Internal wrapper to calculate error."""
        logits = self._forward(X, weights)
        y_reshaped = np.array(y).reshape(-1, 1)
        if self.error_calc == 'BCE':
            preds = np.clip(self.sigmoid(logits), 1e-15, 1 - 1e-15)
            loss = - (y_reshaped * np.log(preds) + (1 - y_reshaped) * np.log(1 - preds))
        else:
            loss = (logits - y_reshaped) ** 2
        return np.mean(loss)

    def _init_weights_gaussian(self, X, y):
        """Internal logic to define solution structure using gaussian distribution."""
        n_features = X.shape[1]
        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        return [
            self.rng.standard_normal((n_features, self.hidden_layer_size)) * 0.1,
            np.zeros(self.hidden_layer_size),
            self.rng.standard_normal((self.hidden_layer_size, n_outputs)) * 0.1,
            np.zeros(n_outputs)
        ]

    def _init_weights_xavier(self, X, y):
        """Internal logic to define solution structure using the xavier distribution."""
        n_features = X.shape[1]
        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        n_hidden = self.hidden_layer_size
        std1 = np.sqrt(2 / (n_features + n_hidden))
        std2 = np.sqrt(2 / (n_hidden + n_outputs))

        return [
            self.rng.normal(0, std1, (n_features, n_hidden)),
            np.zeros(n_hidden),
            self.rng.normal(0, std2, (n_hidden, n_outputs)),
            np.zeros(n_outputs)
        ]

    def fit(self, X_train, y_train):
        """fitting the MLP model."""
        initializer = lambda: (self._init_weights_gaussian(X_train, y_train)
                               if self.weight_init == 'gaussian'
                               else self._init_weights_xavier(X_train, y_train))
        evaluator = lambda w: self._fitness_wrapper(w, X_train, y_train)


        self.optimizer = self.algorithm_class(
            generate_solution=initializer,
            fitness_function=evaluator,
            **self.algo_params
        )

        self.model_weights, _ = self.optimizer.run()
        return self

    def predict_proba(self, X_test):
        """predict the probabilites for X."""
        logits = self._forward(X_test, self.model_weights)
        return self.sigmoid(logits)

    def predict(self, X_test):
        """predict the labels for X."""
        logits = self._forward(X_test, self.model_weights)
        return (logits > 0).astype(int).flatten()

    def score(self, X, y):
        """score the model and return the score."""
        correct = np.sum(self.predict(X) == y)
        return correct / y.shape[0]

    def history(self):
        if hasattr(self, 'optimizer'):
            return self.optimizer.plot_history()
        else:
            raise ValueError("You must call .fit() before accessing history.")

if __name__ == "__main__":

    params_GA={'selection': tournament_selection,
               'crossover': alpha_crossover,
               'mutation': adaptive_mutation,
               'pop_size': 50,
               'n_iter': 1000,
               'mutation_rate': 0.1,
               'maximize': False
               }
    params_DE={'crossover': binomial_crossover,
               'mutation': diff_evol_mutation,
               'pop_size': 50,
               'n_iter': 1000,
               'mutation_factor': 0.9,
               'crossover_rate': 0.8,
               'maximize': False
                }
