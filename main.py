"""
Main entry point for the optimization algorithms comparison.

This module compares Genetic Algorithms, Ant Colony Optimization, and 
Particle Swarm Optimization for solving the Travelling Salesman Problem 
and the Rastrigin Problem.
"""

from dataloader import load_data
from scipy import spatial
from sko.GA import GA_TSP
#from sko.ACO import ACO_TSP
#from sko.PSO import PSO
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_results(best_points, data, optimizer, algorithm_name, dataset_name):
    """Plots the TSP route and the optimization progress."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{algorithm_name} Results for "{dataset_name}" Dataset')

    # Plot the best route
    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = data.values[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    ax[0].set_title('Best Route Found')
    ax[0].set_xlabel('X Coordinate')
    ax[0].set_ylabel('Y Coordinate')

    # Plot the convergence
    ax[1].plot(optimizer.generation_best_Y)
    ax[1].set_title('Convergence Over Generations')
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel('Total Distance')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def run_genetic_algorithm(dataset_name, df, distance_matrix):
    """
    Runs the GA_TSP optimizer on a given dataset.
    """
    print(f'\n--- Running Genetic Algorithm on "{dataset_name}" dataset ---')

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    num_cities = distance_matrix.shape[0]
    print(f"Number of cities: {num_cities}")

    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_cities, size_pop=100, max_iter=4000, prob_mut=0.1)
    best_points, best_distance = ga_tsp.run()

    print(f"GA Optimization complete for '{dataset_name}'!")
    print(f"Best distance: {best_distance}")

    plot_results(best_points, df, ga_tsp, "Genetic Algorithm", dataset_name)

def run_ant_colony_optimization(dataset_name, df, distance_matrix):
    """
    Runs the ACO_TSP optimizer on a given dataset.
    """
    print(f'\n--- Running Ant Colony Optimization on "{dataset_name}" dataset ---')
    print("Note: ACO implementation is a placeholder.")
    
    # TODO: Implement Ant Colony Optimization
    # aco = ACO_TSP(func=..., n_dim=...,)
    # best_points, best_distance = aco.run()
    # plot_results(best_points, df, aco, "Ant Colony", dataset_name)
    pass

def run_particle_swarm_optimization(dataset_name, df, distance_matrix):
    """
    Runs the PSO optimizer on a given dataset.
    Note: Standard PSO is not designed for TSP's discrete permutation problem.
    This is a placeholder for a potential custom or modified PSO implementation.
    """
    print(f'\n--- Running Particle Swarm Optimization on "{dataset_name}" dataset ---')
    print("Note: PSO implementation is a placeholder and not suited for TSP by default.")
    
    # TODO: Implement a suitable version of PSO for TSP if required.
    pass

def main():
    """
    Main function to run and compare optimization algorithms.
    """
    print("Starting optimization algorithms comparison...")
    
    # Load all datasets
    all_data = load_data()
    if not all_data:
        print("No data was loaded. Exiting.")
        return

    # Calculate distance matrices for all datasets
    distance_matrices = {name: spatial.distance.cdist(df.values, df.values, metric='euclidean') for name, df in all_data.items()}

    # --- Define which dataset to run the optimizations on ---
    target_dataset = "medium"
    
    if target_dataset in all_data:
        print(f'\n>>> Running all algorithms on the "{target_dataset}" dataset <<<')
        df = all_data[target_dataset]
        distance_matrix = distance_matrices[target_dataset]

        # Run Genetic Algorithm
        run_genetic_algorithm(target_dataset, df, distance_matrix)

        # Run Ant Colony Optimization
        run_ant_colony_optimization(target_dataset, df, distance_matrix)

        # Run Particle Swarm Optimization
        run_particle_swarm_optimization(target_dataset, df, distance_matrix)
    else:
        print(f'Target dataset "{target_dataset}" not found.')


if __name__ == "__main__":
    main()
