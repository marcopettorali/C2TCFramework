import random
import multiprocessing as mp
from tqdm import tqdm


# ------------------------------------------------------------------------
# 1. A top-level wrapper function that can be pickled
# ------------------------------------------------------------------------
def _fitness_wrapper_global(args):
    """
    Standalone wrapper for fitness evaluation that can be used by multiprocessing.
    Unpacks the args tuple to retrieve:
      - candidate (list/tuple)
      - user fitness function
      - extra positional arguments (fitness_args)
      - extra keyword arguments (fitness_kwargs)
    """
    candidate, fitness_function, fitness_args, fitness_kwargs = args
    return fitness_function(candidate, *fitness_args, **fitness_kwargs)


# ------------------------------------------------------------------------
# 2. The main GA function
# ------------------------------------------------------------------------
def ga_optimization(
    fitness_function,
    vector_size,
    value_type,
    value_range,
    max_fitness=None,
    population_size=50,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism=True,
    tournament_size=3,
    num_processes=None,
    fitness_args=None,
    fitness_kwargs=None,
    tqdm_label="GA Progress",
):
    """
    Performs a Genetic Algorithm optimization using multiprocessing.

    :param fitness_function: A callable that accepts a candidate solution as the
                             first argument, plus any extra arguments, and returns a numeric fitness.
    :param vector_size: The dimension of the candidate solution vector (int).
    :param value_type: The type of each element in the candidate solution, either int or float.
    :param value_range: A tuple (min_value, max_value) describing the allowable range for each element.
    :param population_size: The number of individuals in the population (int).
    :param generations: The number of generations to evolve (int).
    :param crossover_rate: Probability of crossover between pairs of individuals (0 <= float <= 1).
    :param mutation_rate: Probability of mutating each gene (0 <= float <= 1).
    :param elitism: If True, the best individual is carried over each generation (bool).
    :param tournament_size: Size of the tournament for selection (int).
    :param num_processes: Number of worker processes to spawn. By default, uses mp.cpu_count().
    :param fitness_args: A tuple of extra positional arguments for the fitness function.
    :param fitness_kwargs: A dict of extra keyword arguments for the fitness function.
    :return: A dictionary with:
             {
               "best_solution": [list_of_values],
               "best_fitness": float
             }
    """

    if fitness_args is None:
        fitness_args = ()
    if fitness_kwargs is None:
        fitness_kwargs = {}

    # ------------------------------------------------------------------------
    # 2.1. Population Initialization
    # ------------------------------------------------------------------------
    def create_individual():
        """
        Creates a single individual as a list of random values
        (int or float) within the specified value_range.
        """
        if value_type == int:
            return [random.randint(value_range[0], value_range[1]) for _ in range(vector_size)]
        elif value_type == float:
            return [random.uniform(value_range[0], value_range[1]) for _ in range(vector_size)]
        else:
            raise ValueError("value_type must be int or float")

    def create_population(size):
        """
        Creates an entire population (list of individuals).
        """
        return [create_individual() for _ in range(size)]

    population = create_population(population_size)

    # ------------------------------------------------------------------------
    # 2.2. Fitness Evaluation (parallel)
    # ------------------------------------------------------------------------
    def evaluate_population(pop):
        """
        Evaluates the fitness of each individual in 'pop' in parallel,
        returning a list of fitness values.
        """
        with mp.Pool(processes=num_processes) as pool:
            # Build a list of arguments to pass to the global fitness wrapper
            tasks = [(individual, fitness_function, fitness_args, fitness_kwargs) for individual in pop]
            fitness_results = pool.map(_fitness_wrapper_global, tasks)
        return fitness_results

    # ------------------------------------------------------------------------
    # 2.3. Selection (Tournament)
    # ------------------------------------------------------------------------
    def tournament_selection(pop, fits):
        """
        Tournament selection:
        1) Randomly pick 'tournament_size' individuals from pop.
        2) Return the one with the highest fitness.
        """
        selected_indices = random.sample(range(len(pop)), tournament_size)
        best_fit = float("-inf")
        best_ind = None
        for idx in selected_indices:
            if fits[idx] > best_fit:
                best_fit = fits[idx]
                best_ind = pop[idx]
        return best_ind

    # ------------------------------------------------------------------------
    # 2.4. Crossover
    # ------------------------------------------------------------------------
    def crossover(parent1, parent2):
        """
        Performs a 1-point crossover between two parents.
        Returns two offspring.
        """
        if random.random() < crossover_rate:
            point = random.randint(1, vector_size - 1)
            offspring1 = parent1[:point] + parent2[point:]
            offspring2 = parent2[:point] + parent1[point:]
            return offspring1, offspring2
        else:
            return parent1[:], parent2[:]

    # ------------------------------------------------------------------------
    # 2.5. Mutation
    # ------------------------------------------------------------------------
    def mutate(individual):
        """
        Mutates an individual's genes according to 'mutation_rate'.
        """
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                if value_type == int:
                    individual[i] = random.randint(value_range[0], value_range[1])
                else:  # float
                    individual[i] = random.uniform(value_range[0], value_range[1])
        return individual

    # ------------------------------------------------------------------------
    # 2.6. Main Evolution Loop
    # ------------------------------------------------------------------------
    best_individual = None
    best_fitness_val = float("-inf")

    pbar = tqdm(total=generations, desc="GA Progress", unit="gen")
    for generation in range(generations):
        # If the fitness reaches the max_value, stop the algorithm
        if max_fitness is not None and best_fitness_val >= max_fitness:
            print(f"Max fitness reached: {best_fitness_val:.4f}. Stopping evolution.")
            break

        # Evaluate the population in parallel
        fitnesses = evaluate_population(population)

        # Find the best individual in this generation
        current_best_fit = max(fitnesses)
        current_best_idx = fitnesses.index(current_best_fit)
        current_best_ind = population[current_best_idx]

        # Update global best
        if current_best_fit > best_fitness_val:
            best_fitness_val = current_best_fit
            best_individual = current_best_ind[:]

        # best individual string must be at most 40 chars long. If longer, put ellipsis at the middle
        best_individual_str = str(best_individual)
        if len(best_individual_str) > 40:
            best_individual_str = best_individual_str[:20] + "..." + best_individual_str[-20:]

        pbar.set_description(f"{tqdm_label}: best fitness = {best_fitness_val:.4f}, best individual = {best_individual_str}")
        pbar.update(1)

        # Create new population
        new_population = []

        # Elitism: carry over the best individual
        if elitism:
            new_population.append(best_individual[:])

        # Fill up the new population
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            offspring1, offspring2 = crossover(parent1, parent2)

            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)

            new_population.append(offspring1)
            if len(new_population) < population_size:
                new_population.append(offspring2)

        population = new_population

    return {"best_solution": best_individual, "best_fitness": best_fitness_val}


# -----------------------------------------------------------------------------
# USAGE EXAMPLE:
#
# if __name__ == "__main__":
#     # It's important to put your GA call inside this 'if __name__ == "__main__":'
#     # block, especially on Windows, to avoid infinite process spawning.
#
#     def my_fitness_function(candidate, factor, offset):
#         # Example: negative sum of squares, scaled by 'factor', plus an offset
#         return -(sum(x**2 for x in candidate) * factor) + offset
#
#     best = ga_optimization(
#         fitness_function=my_fitness_function,
#         vector_size=5,
#         value_type=float,
#         value_range=(-5, 5),
#         population_size=30,
#         generations=20,
#         crossover_rate=0.7,
#         mutation_rate=0.1,
#         elitism=True,
#         tournament_size=3,
#         num_processes=4,    # or None for auto-detection
#         fitness_args=(0.5, 10.0),
#         fitness_kwargs={}
#     )
#
#     print("Best solution:", best["best_solution"])
#     print("Best fitness:", best["best_fitness"])
# -----------------------------------------------------------------------------
