"""
Main entry point for the optimization algorithms comparison.

This module compares Genetic Algorithms, Ant Colony Optimization, and 
Particle Swarm Optimization for solving the Travelling Salesman Problem 
and the Rastrigin Problem.
"""

from dataloader import load_data
from analysis import run_analysis


def main():
    """
    Main function to run the optimization algorithms comparison.
    """
    print("Starting optimization algorithms comparison...")
    
    # Load data
    data = load_data()
    
    # Run analysis
    results = run_analysis(data)
    
    print("Analysis complete!")
    return results


if __name__ == "__main__":
    main()
