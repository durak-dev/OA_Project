import matplotlib.pyplot as plt
import numpy as np
from FFO_MLP import OptimizationMLP_Classifier


def compare_algorithms(X, y, configs, n_repeats=30):
    """
    configs: list of dictionaries:
             {'class': AlgoClass, 'params': dict, 'name': str}
    """
    plt.figure(figsize=(10, 6))

    for config in configs:
        all_histories = []
        print('---------------------------------------------')
        print(config['class'])
        print('---------------------------------------------')
        for rep in range(n_repeats):
            mlp = OptimizationMLP_Classifier(config['class'], **config['params'])
            mlp.fit(X, y)
            all_histories.append(mlp.optimizer.history)
            print(f"{rep+1} of {n_repeats} runs completed")

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
    plt.savefig(f'figures/algorithm_performance.png')

if __name__ == "__main__":
    pass