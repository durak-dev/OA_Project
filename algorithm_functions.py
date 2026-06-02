import numpy as np
import random as rd



def alpha_crossover(parent1, parent2, rng=None):
    rng = rng if rng is not None else np.random.default_rng()

    child1 = []
    child2 = []
    for index in range(len(parent1)):
        alpha = rng.random()
        child1.append((1 - alpha) * parent1[index] + alpha * parent2[index])
        child2.append((1 - alpha) * parent2[index] + alpha * parent1[index])
    return child1, child2


def swap_crossover(parent1, parent2, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    child1 = []
    child2 = []
    for index in range(len(parent1)):
        if rng.random() < 0.5:
            child1.append(parent1[index])
            child2.append(parent2[index])
        else:
            child1.append(parent2[index])
            child2.append(parent1[index])
    return child1, child2


def binomial_crossover(individual, mutant, cr, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    mutant_child = []
    for i in range(len(individual)):
        if rng.random() < cr:
            mutant_child.append(mutant[i])
        else:
            mutant_child.append(individual[i])
    return mutant_child


def tournament_selection(population, fitnesses, tournament_size, maximize, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    sample_indices = rng.choice(len(population), size=tournament_size, replace=False)

    tournament = [population[i] for i in sample_indices]
    fit_tournament = [fitnesses[i] for i in sample_indices]
    if maximize:
        return tournament[np.argmax(fit_tournament)]
    else:
        return tournament[np.argmin(fit_tournament)]


def individual_mutation(child, mutation_rate, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    for i in range(0, len(child), 2):
        for j in range(len(child[i])):
            if mutation_rate > rng.random():
                child[i][j] *= rng.standard_normal()
    return child

def adaptive_mutation(child, mutation_rate, rng=None):
    """
    Mutates weights with a strength scaled by the layer dimensions
    (using Glorot-style logic).
    """
    rng = rng if rng is not None else np.random.default_rng()
    for i in range(0, len(child), 2):
        weights = child[i]
        bias = child[i + 1]
        fan_in, fan_out = weights.shape
        xavier_scale = np.sqrt(2 / (fan_in + fan_out))
        dynamic_strength = 0.1 * xavier_scale

        mask_w = rng.random(weights.shape) < mutation_rate
        mask_b = rng.random(bias.shape) < mutation_rate
        child[i] += mask_w * rng.normal(0, dynamic_strength, weights.shape)
        child[i + 1] += mask_b * rng.normal(0, dynamic_strength, bias.shape)

    return child


def diff_evol_mutation(individual, population, scaling_rate, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    pop_size = len(population)

    idx_list = []
    while len(idx_list) < 2:
        r = rng.integers(0, pop_size)
        candidate = population[r]
        if candidate is not individual and r not in idx_list:
            idx_list.append(r)

    other1 = population[idx_list[0]]
    other2 = population[idx_list[1]]

    mutant = []
    for index in range(len(individual)):
        mutated_dna = individual[index] + scaling_rate * (other1[index] - other2[index])
        mutated_dna = np.clip(mutated_dna, -5, 5)
        mutant.append(mutated_dna)

    return mutant
