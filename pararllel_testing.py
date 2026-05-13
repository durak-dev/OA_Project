import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from joblib import Parallel, delayed
from FFO_MLP import OptimizationMLP_Classifier

def _run_single_experiment(config, X_train, y_train, X_test, y_test):
    """Helper function to perform a single optimization run."""
    mlp = OptimizationMLP_Classifier(config['class'], 10, **config['params'])
    mlp.fit(X_train, y_train)

    # Prepare history data
    history = [
        {'Iteration': i, 'Loss': loss, 'Algorithm': config['name']}
        for i, loss in enumerate(mlp.optimizer.history)
    ]

    # Prepare score data
    scores = [
        {'Score': mlp.score(X_train, y_train), 'Algorithm': config['name'], 'Type': 'Training'},
        {'Score': mlp.score(X_test, y_test), 'Algorithm': config['name'], 'Type': 'Testing'}
    ]
    
    return history, scores

def compare_algorithms(X_train, y_train, X_test, y_test, configs, n_repeats=30):
    # 1. Parallel Execution
    # n_jobs=-1 utilizes all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(_run_single_experiment)(config, X_train, y_train, X_test, y_test)
        for config in configs
        for _ in range(n_repeats)
    )

    # 2. Flatten the results
    history_data = []
    score_data = []
    for hist, scores in results:
        history_data.extend(hist)
        score_data.extend(scores)

    # 3. Convert to DataFrames
    df_history = pd.DataFrame(history_data)
    df_scores = pd.DataFrame(score_data)

    # 4. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    sns.lineplot(data=df_history, x='Iteration', y='Loss', hue='Algorithm', ax=ax1)
    ax1.set_title('Fitness (Loss) Convergence')

    df_scores['Alg_Group'] = df_scores['Algorithm'] + '\n(' + df_scores['Type'] + ')'

    sns.violinplot(
        data=df_scores,
        x='Alg_Group',
        y='Score',
        hue='Type',
        inner=None,
        dodge=False,
        ax=ax2
    )

    sns.boxplot(
        data=df_scores,
        x='Alg_Group',
        y='Score',
        hue='Type',
        width=0.15,
        dodge=False,
        showfliers=False,
        ax=ax2,
        boxprops={'zorder': 2, 'facecolor': 'none', 'edgecolor': 'black'},
        medianprops={'color': 'black', 'linewidth': 1.5}
    )

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:2], labels[:2], title='Type')
    ax2.set_title(f'Score Distribution: Train vs. Test ({n_repeats} Runs)')

    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    names_str = "_".join([c['name'].replace(" ", "_") for c in configs])
    plt.savefig(f'figures/algorithm_performance{names_str}.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    pass