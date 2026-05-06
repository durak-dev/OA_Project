import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from FFO_MLP import OptimizationMLP_Classifier


def compare_algorithms(X_train, y_train, X_test, y_test, configs, n_repeats=30):
    history_data = []
    score_data = []

    # 1. Collect all data into lists
    for config in configs:
        print(f'Running {config["name"]}...')
        for rep in range(n_repeats):
            mlp = OptimizationMLP_Classifier(config['class'], 10, **config['params'])
            mlp.fit(X_train, y_train)

            # Store History (Long format for Seaborn lineplot)
            for i, loss in enumerate(mlp.optimizer.history):
                history_data.append({'Iteration': i, 'Loss': loss, 'Algorithm': config['name']})

            # Store Score
            score_data.append({'Score': mlp.score(X_train, y_train), 'Algorithm': config['name'], 'Type': 'Training'})
            score_data.append({'Score': mlp.score(X_test, y_test), 'Algorithm': config['name'], 'Type': 'Testing'})

    # 2. Convert to DataFrames
    df_history = pd.DataFrame(history_data)
    df_scores = pd.DataFrame(score_data)

    # 3. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # Plot Fitness (Seaborn handles the mean/std shadow automatically)
    sns.lineplot(data=df_history, x='Iteration', y='Loss', hue='Algorithm', ax=ax1)
    ax1.set_title('Fitness (Loss) Convergence')

    df_scores['Alg_Group'] = df_scores['Algorithm'] + '\n(' + df_scores['Type'] + ')'

    # 1. Plot the Violin (dodge=False forces it to dead-center)
    sns.violinplot(
        data=df_scores,
        x='Alg_Group',  # Use the combined column
        y='Score',
        hue='Type',  # Still color by Training/Testing
        inner=None,
        dodge=False,  # CRITICAL: Disables Seaborn's offset math
        ax=ax2
    )

    # 2. Plot the Box (dodge=False forces it to the exact same dead-center)
    sns.boxplot(
        data=df_scores,
        x='Alg_Group',  # Use the combined column
        y='Score',
        hue='Type',
        width=0.15,  # Now you can make this as thin as you want
        dodge=False,  # CRITICAL: Disables Seaborn's offset math
        showfliers=False,
        ax=ax2,
        boxprops={'zorder': 2, 'facecolor': 'none', 'edgecolor': 'black'},
        medianprops={'color': 'black', 'linewidth': 1.5}
    )

    # Optional: Clean up the legend since we mapped hue twice
    handles, labels = ax2.get_legend_handles_labels()
    # Seaborn sometimes duplicates legend entries when layering, this keeps just the first two
    ax2.legend(handles[:2], labels[:2], title='Type')

    ax2.set_title(f'Score Distribution: Train vs. Test ({n_repeats} Runs)')

    plt.tight_layout()
    names_str = "_".join([c['name'].replace(" ", "_") for c in configs])

    # Ensure the figures directory exists before saving
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    names_str = "_".join([c['name'].replace(" ", "_") for c in configs])
    plt.savefig(f'figures/algorithm_performance{names_str}.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    pass