import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from FFO_MLP import OptimizationMLP_Classifier

def compare_algorithms_serial(X_train, y_train, X_test, y_test, configs, n_repeats=30):
    history_data = []
    score_data = []

    # 1. Collect all data into lists
    for config in configs:
        print(f'Running {config["name"]}...')
        for rep in range(n_repeats):
            mlp = OptimizationMLP_Classifier(config['class'], 10, **config['params'])
            mlp.fit(X_train, y_train)

            # Store History
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
    return df_history, df_scores.drop(columns=['Alg_Group'])

def _run_single_experiment(config, X_train, y_train, X_test, y_test):
    """Helper function to perform a single optimization run."""
    mlp = OptimizationMLP_Classifier(config['class'], 10, weight_init='glorot', error_calc='BCE', **config['params'])
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

def compare_algorithms_parallel(X_train, y_train, X_test, y_test, configs, n_repeats=30):
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
    df_scores['Alg_Group'] = df_scores['Algorithm'] + '\n(' + df_scores['Type'] + ')'
    # 4. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    sns.lineplot(data=df_history, x='Iteration', y='Loss', hue='Algorithm', ax=ax1)
    ax1.set_title('Fitness (Loss) Convergence')

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
    return df_history, df_scores.drop(columns=['Alg_Group'])

def run_tukey_hsd(df, grouping_col = 'Algorithm', value_col = 'Score', alpha=0.05):
    """
    Performs a Tukey HSD post-hoc test to compare pairwise differences.
    """
    print(f"\n----------- Tukey HSD Results for {value_col} (alpha = {alpha}) -----------")

    tukey_result = pairwise_tukeyhsd(endog=df[value_col],
                                     groups=df[grouping_col],
                                     alpha=alpha)

    print(tukey_result)
    return tukey_result


def plot_tukey_distributions(df, grouping_col = 'Algorithm', value_col = 'Score', alpha=0.05):
    """
    Plots the Studentized Range ($q$) Distribution for each pairwise comparison
    in the Tukey HSD test, highlighting the rejection regions and observed values.
    """
    # 1. Calculate basic degrees of freedom and MSE
    grouped = df.groupby(grouping_col)[value_col]
    group_means = grouped.mean()
    group_sizes = grouped.count()

    num_groups = len(group_means)
    total_samples = len(df)

    df_between = num_groups - 1
    df_within = total_samples - num_groups

    # Calculate Mean Squared Error (MSE) from ANOVA
    sum_sq_within = sum(grouped.apply(lambda x: np.sum((x - x.mean()) ** 2)))
    mse = sum_sq_within / df_within

    # 2. Find Critical q-value using Studentized Range (using scipy.stats.studentized_range)
    # ppf requires: (1-alpha, number of groups, degrees of freedom within)
    q_critical = stats.studentized_range.ppf(1 - alpha, num_groups, df_within)

    # 3. Evaluate pairs
    algorithms = list(group_means.index)
    pair_count = 1

    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            algo1, algo2 = algorithms[i], algorithms[j]

            # Calculate Observed q-statistic
            mean_diff = abs(group_means[algo1] - group_means[algo2])
            # Standard error for equal group sizes (30 runs each)
            n_group = group_sizes[algo1]
            standard_error = np.sqrt(mse / n_group)
            q_observed = mean_diff / standard_error

            # 4. Plot the q-distribution for this pair
            max_x = max(q_critical, q_observed) * 1.4
            x = np.linspace(0, max_x, 1000)
            y = stats.studentized_range.pdf(x, num_groups, df_within)

            plt.figure(figsize=(9, 5))
            plt.plot(x, y, 'b-', lw=2, label=f'Studentized Range ($q$) Dist ($k={num_groups}, df={df_within}$)')

            # Shade Rejection Region
            x_reject = np.linspace(q_critical, max_x, 500)
            y_reject = stats.studentized_range.pdf(x_reject, num_groups, df_within)
            plt.fill_between(x_reject, 0, y_reject, color='red', alpha=0.3,
                             label=f'Rejection Region ($\\alpha={alpha}$)\n$q_{{crit}} = {q_critical:.2f}$')

            # Plot Observed q-value
            plt.axvline(x=q_observed, color='darkgreen', linestyle='--', lw=2.5,
                        label=f'Observed $q$ ({algo1} vs {algo2}) = {q_observed:.2f}')

            # Annotate decision
            is_rejected = q_observed > q_critical
            status_text = "Reject H0: Significant Difference" if is_rejected else "Fail to Reject H0: No Difference"
            status_color = "darkgreen" if is_rejected else "blue"

            plt.text(max_x * 0.05, max(y) * 0.8, status_text, color=status_color, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            plt.title(f'Tukey HSD Pairwise Distribution: {algo1} vs {algo2}')
            plt.xlabel('$q$ Value (Studentized Range Metric)')
            plt.ylabel('Probability Density')
            plt.xlim(0, max_x)
            plt.ylim(0, max(y) * 1.1)
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)

            filename = f'tukey_q_dist_{algo1}_vs_{algo2}.png'.lower().replace(" ", "_")
            plt.tight_layout()
            plt.savefig('figures/'+filename)
            plt.close()

            print(f"Saved Tukey distribution plot for {algo1} vs {algo2} to {filename}")


if __name__ == "__main__":
    scores = pd.read_csv('data/algorithm_performance_scores.csv', index_col=0)
    print(scores.columns)
    test_scores = scores.drop(columns='Alg_Group')
    test_scores = test_scores[test_scores['Type']=='Testing']
    print(test_scores)

    pass