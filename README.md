# Parkinson's Disease Classification Project

This repository implements machine learning and optimization algorithms to analyze preprocessed biomedical voice measurements for classifying Parkinson's disease. The project evaluates classical metaheuristic optimization techniques alongside neural network structures to maximize classification performance.

---

## Repository Structure

```
OA_Project/
├── main.py                     # Project entry point orchestrating execution
├── FFO_MLP.py                  # Fruit Fly Optimization coupled with Multilayer Perceptron
├── algorithms.py               # Core metaheuristic/optimization algorithms
├── algorithm_functions.py      # Fitness functions and cost evaluations
├── report_generation.py        # Automated reporting and visualization utilities
├── testing.py                  # Script validation and algorithmic testing
├── parkinsons_preprocessed.csv # Cleaned biomedical dataset
└── Project_OA.pdf              # Academic report documenting findings

```

---

## Core Components

### 1. Data Analytics (`main.py`, `parkinsons_preprocessed.csv`)

Loads and executes the workflow utilizing voice frequency characteristics and dysphonia measures. The architecture runs the data pipeline through feature evaluations to distinguish healthy individuals from those with Parkinson's.

### 2. Optimization Implementations (`algorithms.py`, `FFO_MLP.py`)

* **Fruit Fly Optimization Algorithm (FOA / FFO):** Tweaks hyperparameter weights and biases in custom Multilayer Perceptron (`FFO_MLP`) variants to escape local minima.
* **Metaheuristic Variants:** Standardized algorithmic baselines to compare convergence curves against classical gradient descent approaches.

### 3. Reporting and Testing (`report_generation.py`, `testing.py`)

* Generates analytical metric summaries, accuracy trends, and loss curves.
* Enforces test validations on optimization runs to verify code stability across multiple random states.

---

## Getting Started

### Prerequisites

* Python 3.12
* Standard Data Science stack: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

### Execution

Run the baseline optimization configuration and evaluation pipeline:

```bash
python main.py

```

To run individual module validation profiles:

```bash
python testing.py

```
