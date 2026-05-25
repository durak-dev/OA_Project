import matplotlib.pyplot as plt
from algorithm_functions import *

class HillClimbing():
    def __init__(self, val, neighborhood, fitness_function, optimum = 'max'):
        self.val = val
        self.neighborhood = neighborhood
        self.fitness_function = fitness_function
        self.optimum = optimum

    def find_best_neighbor(self, best_sol):
        sols = self.neighborhood(best_sol)
        sols_fit = [self.fitness_function(i) for i in sols]
        bn = None
        if self.optimum == 'max':
            bn = sols[np.argmax(sols_fit)]
        elif self.optimum == 'min':
            bn = sols[np.argmin(sols_fit)]
        return bn

    def run(self):
        best_sol = self.val
        better_neighbor = self.find_best_neighbor(best_sol)
        print(f'Starting Hill Climbing at {self.val}')
        if self.optimum == 'max':
            while self.fitness_function(best_sol) <= self.fitness_function(better_neighbor):
                print(f"current = {best_sol}, best neighbor = {better_neighbor}")
                print(f"fitness current = {self.fitness_function(best_sol)}, fitness neighbor = {self.fitness_function(better_neighbor)}")
                best_sol = better_neighbor
                better_neighbor = self.find_best_neighbor(best_sol)
        elif self.optimum == 'min':
            while self.fitness_function(best_sol) >= self.fitness_function(better_neighbor):
                print(f"current = {best_sol}, best neighbor = {better_neighbor}")
                print(f"fitness current = {self.fitness_function(best_sol)}, fitness neighbor = {self.fitness_function(better_neighbor)}")
                best_sol = better_neighbor
                better_neighbor = self.find_best_neighbor(best_sol)
        return best_sol, self.fitness_function(best_sol)

class SimulatedAnnealing():
    def __init__(self, generate_solution, fitness, generate_neighbor, T=1000, alpha = 0.99, n_iter = 1000, minimize=False):
        self.curr_sol = generate_solution()
        self.fitness = fitness
        self.generate_neighbor = generate_neighbor
        self.T = T
        self.alpha = alpha
        self.n_iter = n_iter
        self.minimize = minimize
        self.history_fit = []
    def run(self):
        best_sol = self.curr_sol
        while self.n_iter > 0:
            self.history_fit.append(self.fitness(best_sol))
            curr_sol_fit = self.fitness(self.curr_sol)
            neighbor = self.generate_neighbor(self.curr_sol)[np.random.randint(0, 49)]
            fitness_neighbor = self.fitness(neighbor)
            if self.minimize and fitness_neighbor < curr_sol_fit:
                self.curr_sol = neighbor

            elif self.minimize==False and fitness_neighbor > curr_sol_fit:
                self.curr_sol = neighbor

            else:
                delta = abs(fitness_neighbor-curr_sol_fit)
                p = np.exp(-(delta/self.T))
                u = np.random.random()
                if u<p:
                    self.curr_sol = neighbor

            if self.fitness(self.curr_sol) > self.fitness(best_sol):
                best_sol = neighbor

            self.T = self.alpha * self.T
            self.n_iter -= 1

        return best_sol, self.fitness(best_sol)
    def plot_history(self):
        print(self.run())
        plt.plot(self.history_fit)
        plt.show()


class GeneticAlgorithm():
    def __init__(self, generate_solution, fitness_function, selection, crossover, mutation,
                 pop_size, n_iter, tournament_size=3, mutation_rate=0.05, maximize=True, elitism=False,
                 random_state=None):
        self.rng = np.random.default_rng(random_state)
        self.population = [generate_solution() for _ in range(pop_size)]
        self.fitness_function = fitness_function
        self.pop_fitness = [self.fitness_function(dna) for dna in self.population]
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.n_iter = n_iter
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.maximize = maximize
        self.elitism = elitism
        self.history = []

    def run(self):
        epoch = self.n_iter
        if self.maximize:
            best_sol = self.population[np.argmax(self.pop_fitness)]
            best_sol_fitness = self.pop_fitness[np.argmax(self.pop_fitness)]
        else:
            best_sol = self.population[np.argmin(self.pop_fitness)]
            best_sol_fitness = self.pop_fitness[np.argmin(self.pop_fitness)]

        while epoch > 0:
            nxt_pop_dna = []
            best_from_prev = best_sol
            while len(nxt_pop_dna) < len(self.population):
                parent1 = self.selection(self.population, self.pop_fitness, tournament_size=self.tournament_size,
                                         maximize=self.maximize, rng=self.rng)
                parent2 = self.selection(self.population, self.pop_fitness, tournament_size=self.tournament_size,
                                         maximize=self.maximize, rng=self.rng)
                child1, child2 = self.crossover(parent1, parent2, rng=self.rng)
                nxt_pop_dna.append(self.mutation(child1, self.mutation_rate, rng=self.rng))
                nxt_pop_dna.append(self.mutation(child2, self.mutation_rate, rng=self.rng))

            if self.elitism:
                nxt_pop_dna[self.rng.integers(0, len(self.population))] = best_from_prev

            self.population = nxt_pop_dna
            self.pop_fitness = [self.fitness_function(kid) for kid in self.population]
            epoch -= 1

            if self.maximize:
                index_best_fit = np.argmax(self.pop_fitness)
                new_sol_fit = self.pop_fitness[index_best_fit]
                if new_sol_fit > best_sol_fitness:
                    best_sol = self.population[index_best_fit]
                    best_sol_fitness = new_sol_fit
            else:
                index_best_fit = np.argmin(self.pop_fitness)
                new_sol_fit = self.pop_fitness[index_best_fit]
                if new_sol_fit < best_sol_fitness:
                    best_sol = self.population[index_best_fit]
                    best_sol_fitness = new_sol_fit
            self.history.append(best_sol_fitness)
        return best_sol, best_sol_fitness

    def plot_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history)
        plt.title('Genetic Algorithm Performance')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.subplots_adjust(bottom=0.2)
        params = (f"Pop Size: {len(self.population)} | Iterations: {self.n_iter} | "
                  f"Mutation Factor: {self.mutation_rate}")
        plt.figtext(0.5, 0.05, params, ha="center", fontsize=10,
                    bbox={"facecolor": "lightblue", "alpha": 0.2, "pad": 5})
        plt.savefig('figures/GA_optimization_history.png')
        plt.show()


class DifferentialEvolution():
    def __init__(self, generate_solution, fitness_function, crossover, mutation,
                 pop_size, n_iter, crossover_rate=0.7, mutation_factor=0.5, maximize=True, random_state=None):

        # NEW: Initialize the random generator
        self.rng = np.random.default_rng(random_state)

        self.population = [generate_solution() for _ in range(pop_size)]
        self.fitness_function = fitness_function
        self.pop_fitness = [self.fitness_function(dna) for dna in self.population]
        self.crossover = crossover
        self.mutation = mutation
        self.n_iter = n_iter
        self.crossover_rate = crossover_rate
        self.mutation_factor = mutation_factor
        self.maximize = maximize
        self.history = []

    def run(self):
        epoch = self.n_iter
        if self.maximize:
            best_sol = self.population[np.argmax(self.pop_fitness)]
            best_sol_fitness = self.pop_fitness[np.argmax(self.pop_fitness)]
        else:
            best_sol = self.population[np.argmin(self.pop_fitness)]
            best_sol_fitness = self.pop_fitness[np.argmin(self.pop_fitness)]

        while epoch > 0:
            for i in range(len(self.population)):
                individual = self.population[i]
                individual_fitness = self.pop_fitness[i]

                # NEW: Passed rng=self.rng to external mutation and crossover functions
                mutant = self.mutation(individual, self.population, self.mutation_factor, rng=self.rng)
                mutant_child = self.crossover(individual, mutant, self.crossover_rate, rng=self.rng)
                mutant_child_fitness = self.fitness_function(mutant_child)

                if self.maximize:
                    mutant_better = mutant_child_fitness >= individual_fitness
                else:
                    mutant_better = mutant_child_fitness <= individual_fitness

                if mutant_better:
                    self.population[i] = mutant_child
                    self.pop_fitness[i] = mutant_child_fitness

            epoch -= 1

            if self.maximize:
                index_best_fit = np.argmax(self.pop_fitness)
                new_sol_fit = self.pop_fitness[index_best_fit]
                if new_sol_fit > best_sol_fitness:
                    best_sol = self.population[index_best_fit]
                    best_sol_fitness = new_sol_fit
            else:
                index_best_fit = np.argmin(self.pop_fitness)
                new_sol_fit = self.pop_fitness[index_best_fit]
                if new_sol_fit < best_sol_fitness:
                    best_sol = self.population[index_best_fit]
                    best_sol_fitness = new_sol_fit
            self.history.append(best_sol_fitness)
        return best_sol, best_sol_fitness

    def plot_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history)
        plt.title('Differential Evolution Performance')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.subplots_adjust(bottom=0.2)
        params = (f"Pop Size: {len(self.population)} | Iterations: {self.n_iter} | "
                  f"Mutation Factor: {self.mutation_factor}")
        plt.figtext(0.5, 0.05, params, ha="center", fontsize=10,
                    bbox={"facecolor": "lightblue", "alpha": 0.2, "pad": 5})
        plt.savefig('figures/DE_optimization_history.png')
        plt.show()


class ParticleSwarm():
    def __init__(self, generate_solution, fitness_function, pop_size, n_iter,
                 w=0.7, c1=1.5, c2=1.5, maximize=True, v_max=0.1, w_limit=3.0, random_state=None):

        self.rng = np.random.default_rng(random_state)

        self.generate_solution = generate_solution
        self.fitness_function = fitness_function
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.maximize = maximize

        self.v_max = v_max
        self.w_limit = w_limit

        self.positions = [generate_solution() for _ in range(pop_size)]
        self.velocities = [self.init_velocity(pos) for pos in self.positions]
        self.pbest = [[arr.copy() for arr in pos] for pos in self.positions]
        self.pbest_fitness = [self.fitness_function(pos) for pos in self.positions]

        if self.maximize:
            best_idx = np.argmax(self.pbest_fitness)
        else:
            best_idx = np.argmin(self.pbest_fitness)

        self.gbest = [arr.copy() for arr in self.positions[best_idx]]
        self.gbest_fitness = self.pbest_fitness[best_idx]
        self.history = []

    def init_velocity(self, pos):
        vel = []
        for p in pos:
            vel.append(self.rng.uniform(-self.v_max, self.v_max, p.shape))
        return vel

    def update_velocity(self, vel, pos, pbest, gbest):
        new_vel = []
        for i in range(len(vel)):
            r1 = self.rng.random(vel[i].shape)
            r2 = self.rng.random(vel[i].shape)
            cognitive = self.c1 * r1 * (pbest[i] - pos[i])
            social = self.c2 * r2 * (gbest[i] - pos[i])

            v = self.w * vel[i] + cognitive + social
            new_vel.append(np.clip(v, -self.v_max, self.v_max))
        return new_vel

    def update_position(self, pos, vel):
        new_pos = []
        for i in range(len(pos)):
            updated_p = pos[i] + vel[i]
            new_pos.append(np.clip(updated_p, -self.w_limit, self.w_limit))
        return new_pos

    def run(self):
        for _ in range(self.n_iter):
            for i in range(self.pop_size):
                self.velocities[i] = self.update_velocity(
                    self.velocities[i], self.positions[i], self.pbest[i], self.gbest
                )

                self.positions[i] = self.update_position(self.positions[i], self.velocities[i])

                fitness = self.fitness_function(self.positions[i])

                update_pbest = (self.maximize and fitness > self.pbest_fitness[i]) or \
                               (not self.maximize and fitness < self.pbest_fitness[i])

                if update_pbest:
                    self.pbest[i] = [arr.copy() for arr in self.positions[i]]
                    self.pbest_fitness[i] = fitness

                    update_gbest = (self.maximize and fitness > self.gbest_fitness) or \
                                   (not self.maximize and fitness < self.gbest_fitness)
                    if update_gbest:
                        self.gbest = [arr.copy() for arr in self.positions[i]]
                        self.gbest_fitness = fitness

            self.history.append(self.gbest_fitness)
        return self.gbest, self.gbest_fitness


if __name__ == "__main__":
    pass