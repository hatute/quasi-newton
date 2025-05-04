import os
import time
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import psutil

# Import our implementations
from bfgs import BFGS
from gradient_descent import GradientDescent
from lbfgs import LBFGS, MemoryTracker
from newton import Newton
from plotting import (
    plot_2d_contour,
    plot_all_comparisons,
    plot_individual_contours,
    plot_loss_trends,
    plot_memory_usage,
    plot_time_vs_loss,
)


# Define test functions
def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2"""
    a, b = 1.0, 100.0
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


def rosenbrock_grad(x):
    """Gradient of Rosenbrock function"""
    a, b = 1.0, 100.0
    dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    dy = 2 * b * (x[1] - x[0]**2)
    return np.array([dx, dy])


def rosenbrock_hess(x):
    """Hessian of Rosenbrock function"""
    a, b = 1.0, 100.0
    dxx = 2 - 4 * b * (x[1] - 3 * x[0]**2)
    dxy = -4 * b * x[0]
    dyy = 2 * b
    return np.array([[dxx, dxy], [dxy, dyy]])


def quadratic(x):
    """Simple quadratic function: f(x,y) = x^2 + 10*y^2"""
    return x[0]**2 + 10 * x[1]**2


def quadratic_grad(x):
    """Gradient of quadratic function"""
    return np.array([2 * x[0], 20 * x[1]])


def quadratic_hess(x):
    """Hessian of quadratic function"""
    return np.array([[2, 0], [0, 20]])


def run_optimization_with_timing_and_memory(
    optimizer_class, 
    args: tuple, 
    x0: np.ndarray,
    track_memory: bool = True
) -> Dict:
    """Run optimization while tracking time and memory."""
    memory_tracker = MemoryTracker() if track_memory else None
    
    # Start timing
    start_time = time.time()
    time_history = []
    
    # Create optimizer
    optimizer = optimizer_class(*args)
    
    # Start memory tracking
    if track_memory:
        memory_tracker.start()
    
    # Run optimization
    time_history.append(0.0)
    x_opt, f_opt, history_x, history_f = optimizer.optimize(x0, return_history=True)
    
    # Record final time
    current_time = time.time() - start_time
    
    # Create proper time history (one entry per iteration)
    time_history = np.linspace(0, current_time, len(history_x)).tolist()
    
    # Record final memory if tracking
    if track_memory:
        memory_tracker.record()
        # Extend memory usage to match history length
        memory_history = memory_tracker.get_usage()
        if len(memory_history) < len(history_x):
            memory_history.extend([memory_history[-1]] * (len(history_x) - len(memory_history)))
    
    # Create result dictionary
    result = {
        'x_opt': x_opt,
        'f_opt': f_opt,
        'history_x': history_x,
        'history_f': history_f,
        'time_history': time_history,
        'iterations': len(history_x) - 1
    }
    
    if track_memory:
        result['memory_usage'] = memory_history
    
    return result


def main():
    """Main function to run all optimizations and create plots."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test function to use (Rosenbrock)
    f = rosenbrock
    grad_f = rosenbrock_grad
    hess_f = rosenbrock_hess
    
    # Starting point
    x0 = np.array([-3.0, 4.0])
    
    print("Running optimizations on Rosenbrock function...")
    print(f"Starting point: {x0}")
    print("-" * 50)
    
    # Run optimizations
    results = {}
    
    # Newton's method
    print("Running Newton's method...")
    results['Newton'] = run_optimization_with_timing_and_memory(
        Newton, 
        (f, grad_f, hess_f),
        x0
    )
    
    # BFGS
    print("Running BFGS...")
    results['BFGS'] = run_optimization_with_timing_and_memory(
        BFGS,
        (f, grad_f),
        x0
    )
    
    # L-BFGS
    print("Running L-BFGS...")
    results['L-BFGS'] = run_optimization_with_timing_and_memory(
        LBFGS,
        (f, grad_f),
        x0
    )
    
    # Gradient Descent
    print("Running Gradient Descent...")
    results['Gradient Descent'] = run_optimization_with_timing_and_memory(
        GradientDescent,
        (f, grad_f),
        x0,
        track_memory=False  # GD is simple enough that memory tracking isn't interesting
    )
    
    # Print results
    print("\nOptimization Results:")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Optimal point: {result['x_opt']}")
        print(f"  Optimal value: {result['f_opt']:.6f}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Time taken: {result['time_history'][-1]:.4f} seconds")
        if 'memory_usage' in result:
            print(f"  Memory usage: {result['memory_usage'][-1]:.2f} MB")
        print()
    
    # Create plots
    print("Creating plots...")
    
    # 1. Individual contour plots
    fig = plt.figure(figsize=(16, 12))
    for i, (name, result) in enumerate(results.items(), 1):
        ax = fig.add_subplot(2, 2, i)
    plt.tight_layout()
    plt.savefig("individual_contours.png", dpi=300)
    plt.close()
    
    # 2. Combined contour plot
    fig = plt.figure(figsize=(10, 8))
    paths = {name: result['history_x'] for name, result in results.items()}
    plot_2d_contour(f, (-3.5, 1), (-1, 5), paths,
                   title="All Optimizers on Rosenbrock Function")
    plt.savefig("combined_contour.png", dpi=300)
    plt.close()
    
    # 3. Loss trends
    fig = plt.figure(figsize=(10, 6))
    losses = {name: result['history_f'] for name, result in results.items()}
    plot_loss_trends(losses, title="Loss Trends Comparison", log_scale=True)
    plt.savefig("loss_trends.png", dpi=300)
    plt.close()
    
    # 4. Time vs Loss
    fig = plt.figure(figsize=(10, 6))
    time_loss = {name: (result['time_history'], result['history_f']) 
                 for name, result in results.items()}
    plot_time_vs_loss(time_loss, title="Running Time vs Loss", log_scale=True)
    plt.savefig("time_vs_loss.png", dpi=300)
    plt.close()
    
    # 5. Memory usage (excluding Gradient Descent)
    fig = plt.figure(figsize=(10, 6))
    memory = {name: result['memory_usage'] 
              for name, result in results.items() 
              if 'memory_usage' in result}
    plot_memory_usage(memory, title="Memory Usage Comparison")
    plt.savefig("memory_usage.png", dpi=300)
    plt.close()
    
    # 6. All comparisons in one figure
    plot_all_comparisons(results, f, (-3.5, 1), (-1, 5), 
                        save_path="all_comparisons.png")
    
    print("All plots have been saved!")
    
    # Additional test with a simpler quadratic function for comparison
    print("\nTesting with a simpler quadratic function for comparison...")
    
    # Define simple quadratic function
    def quadratic(x):
        """Simple quadratic function: f(x,y) = x^2 + y^2"""
        return x[0]**2 + x[1]**2
    
    def quadratic_grad(x):
        """Gradient of quadratic function"""
        return np.array([2 * x[0], 2 * x[1]])
    
    def quadratic_hess(x):
        """Hessian of quadratic function"""
        return np.array([[2, 0], [0, 2]])
    
    # Run on quadratic function
    x0_quad = np.array([5.0, 5.0])
    results_quad = {}
    
    # Newton
    results_quad['Newton'] = run_optimization_with_timing_and_memory(
        Newton, 
        (quadratic, quadratic_grad, quadratic_hess),
        x0_quad,
        track_memory=False
    )
    
    # BFGS
    results_quad['BFGS'] = run_optimization_with_timing_and_memory(
        BFGS,
        (quadratic, quadratic_grad),
        x0_quad,
        track_memory=False
    )
    
    # L-BFGS
    results_quad['L-BFGS'] = run_optimization_with_timing_and_memory(
        LBFGS,
        (quadratic, quadratic_grad),
        x0_quad,
        track_memory=False
    )
    
    # Create comparison plot for quadratic function
    fig = plt.figure(figsize=(10, 8))
    paths_quad = {name: result['history_x'] for name, result in results_quad.items()}
    plot_2d_contour(quadratic, (-6, 6), (-6, 6), paths_quad,
                   title="Optimizers on Quadratic Function")
    plt.savefig("quadratic_comparison.png", dpi=300)
    plt.close()
    
    print("All optimizations completed successfully!")


if __name__ == "__main__":
    main()