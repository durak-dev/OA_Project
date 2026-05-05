import numpy as np
import random as rd

def alpha_crossover(parent1, parent2):
    child1 = []
    child2 = []
    for index in range(len(parent1)):
        alpha = np.random.rand()
        child1.append((1-alpha)*parent1[index] + alpha*parent2[index])
        child2.append((1-alpha)*parent2[index] + alpha*parent1[index])
    return child1, child2

def swap_crossover(parent1, parent2):
    child1 = []
    child2 = []
    for index in range(len(parent1)):
        if np.random.rand() < 0.5:
            child1.append(parent1[index])
            child2.append(parent2[index])
        else:
            child1.append(parent2[index])
            child2.append(parent1[index])
    return child1, child2

def binomial_crossover(individual, mutant, cr):
    mutant_child = []
    for i in range(len(individual)):
        if np.random.rand() < cr:
            mutant_child.append(mutant[i])
        else:
            mutant_child.append(individual[i])
    return mutant_child

def tournament_selection(population, fitnesses, tournament_size, maximize):
    sample_indices = np.random.choice([i for i in range(len(population))], size=tournament_size, replace=False)
    tournament = [population[i] for i in sample_indices]
    fit_tournament = [fitnesses[i] for i in sample_indices]
    if maximize:
        return tournament[np.argmax(fit_tournament)]
    else:
        return tournament[np.argmin(fit_tournament)]

def individual_mutation(child, mutation_rate):
    for i in range(0, len(child), 2):
        for j in range(len(child[i])):
            if mutation_rate > np.random.rand():
                child[i][j] *= np.random.randn()

def adaptive_mutation(child, mutation_rate):
    """
    Mutates weights with a strength scaled by the layer dimensions
    (using Glorot-style logic).
    """
    for i in range(0, len(child), 2):
        weights = child[i]
        bias = child[i + 1]
        fan_in, fan_out = weights.shape
        # Xavier/Glorot scaling factor
        # This keeps the variance of the 'nudge' proportional to layer size
        xavier_scale = np.sqrt(2 / (fan_in + fan_out))
        # Calculate dynamic strength
        dynamic_strength = 0.1 * xavier_scale
        # Perform mutation
        mask_w = np.random.rand(*weights.shape) < mutation_rate
        mask_b = np.random.rand(*bias.shape) < mutation_rate
        child[i] += mask_w * np.random.normal(0, dynamic_strength, weights.shape)
        child[i + 1] += mask_b * np.random.normal(0, dynamic_strength, bias.shape)

    return child

def diff_evol_mutation(individual, population, scaling_rate):
    mutant = []
    others = rd.sample(population, 2)
    other1 = others[0]
    other2 = others[1]
    for index in range(len(individual)):
        mutated_dna = individual[index] + scaling_rate * (other1[index] - other2[index])
        mutated_dna = np.clip(mutated_dna, -5, 5)
        mutant.append(mutated_dna)
    return mutant


