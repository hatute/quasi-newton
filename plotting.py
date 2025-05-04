import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

import bfgs as bfgs
import lbfgs as lbfgs
import newton as newton


def plot_contour_2d(
    history_x,
    f_contour,
    f_name,
    x_range=(-2, 2),
    y_range=(-1, 3),
    figsize=(12, 8),
    save=True,
):
    """
    Plot optimization path on Rosenbrock function contour plot.

    Args:
        history_x: List of points visited during optimization
        history_f: List of function values during optimization
        x_range: Tuple of (min, max) for x-axis
        y_range: Tuple of (min, max) for y-axis
        levels: Contour levels for the plot
        figsize: Figure size as (width, height)

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    levels = np.logspace(0, 1.5, 20)
    # Create a grid for the contour plot
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_grid = np.linspace(x_min, x_max, 400)
    y_grid = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Calculate function values for the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f_contour(np.array([X[i, j], Y[i, j]]))

    # Create grayscale contour plot
    contour = ax.contour(
        X, Y, Z, levels=levels, colors="black", linewidths=0.5, alpha=0.7
    )
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap="gray_r", alpha=0.8)
    plt.colorbar(contourf, label="Function Value")

    # Plot the optimization path
    history_x = np.array(history_x)
    ax.plot(
        1,
        1,
        "y*",
        markersize=18,
        alpha=1.0,
        markeredgewidth=0.1,
        label="True Minimum",
    )
    ax.plot(
        history_x[:, 0],
        history_x[:, 1],
        "r.--",
        linewidth=1,
        markersize=5,
        alpha=0.7,
        label=f"{f_name} Path ({len(history_x) - 1} iterations)",
    )
    ax.plot(history_x[0, 0], history_x[0, 1], "go", markersize=6, label="Start Point")
    ax.plot(history_x[-1, 0], history_x[-1, 1], "ro", markersize=6, label="End Point")

    # Set axis properties
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    plt.title(f"{f_name} Optimization on Rosenbrock Function", fontsize=16)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")

    # Set aspect ratio
    ax.set_aspect("auto")

    # Keep all spines visible
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.tight_layout()
    if save:
        save_path = f"./figures/{f_name}_2dcontour.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_loss_comparison(
    results: dict,
    title: str = "Optimization Convergence Comparison",
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    log_scale: bool = False,
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None,
    show_convergence_rate: bool = True,
) -> None:
    """
    Advanced plotting function with additional features like convergence rate visualization.

    Parameters:
    -----------
    results : dict
        Dictionary with keys as algorithm names and values as optimization results
        Example: {'BFGS': bfgs_result, 'Newton': newton_result, 'L-BFGS': lbfgs_result}
    """
    sns.set_style("whitegrid")

    if show_convergence_rate:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    colors = {"BFGS": "blue", "Newton": "red", "L-BFGS": "green"}
    markers = {"BFGS": "o", "Newton": "s", "L-BFGS": "^"}

    # Main loss plot
    for name, result in results.items():
        _, _, _, loss_history = result
        iterations = list(range(len(loss_history)))

        ax1.plot(
            iterations,
            loss_history,
            label=name,
            marker=markers.get(name, "o"),
            markersize=4,
            linewidth=1,
            color=colors.get(name, "black"),
            alpha=0.8,
        )

    ax1.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax1.set_ylabel(ylabel, fontsize=10)
    ax1.legend(fontsize=12, loc="upper right")
    ax1.grid(True, alpha=0.3)

    if log_scale:
        ax1.set_yscale("log")
    ax1.text(
        -0.05,
        1.05,
        "A",
        transform=ax1.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # Remove x-axis label from the top plot
    # if show_convergence_rate:
    #     ax1.set_xticklabels([])
    # else:
    #     ax1.set_xlabel(xlabel, fontsize=14)

    # Convergence rate plot
    if show_convergence_rate:
        for name, result in results.items():
            _, _, _, loss_history = result
            if len(loss_history) > 1:
                rates = []
                for i in range(1, len(loss_history)):
                    if loss_history[i - 1] != 0:
                        rate = abs(loss_history[i] - loss_history[i - 1]) / abs(
                            loss_history[i - 1]
                        )
                        rates.append(rate)
                    else:
                        rates.append(0)

                iterations = list(range(1, len(loss_history)))
                ax2.plot(
                    iterations,
                    rates,
                    label=name,
                    marker=markers.get(name, "o"),
                    markersize=3,
                    linewidth=1,
                    color=colors.get(name, "black"),
                    alpha=0.7,
                )

        ax2.set_xlabel(xlabel, fontsize=14)
        ax2.set_ylabel("Relative Change", fontsize=10)
        ax2.set_yscale("log")
        ax2.legend(fontsize=10, loc="lower right")
        ax2.grid(True, alpha=0.3)
        ax2.text(
            -0.05,
            1.05,
            "B",
            transform=ax2.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="right",
        )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_quadratic_problem(n_dim: int) -> Tuple[callable, callable, callable]:
    """
    Create a quadratic optimization problem in n dimensions.
    f(x) = 0.5 * x^T * Q * x + b^T * x

    Args:
        n_dim: Dimension of the problem

    Returns:
        f: Objective function
        grad_f: Gradient function
        hess_f: Hessian function
    """
    # Generate a random positive definite matrix Q
    A = np.random.randn(n_dim, n_dim)
    Q = A.T @ A + 0.1 * np.eye(n_dim)  # Ensure positive definiteness
    b = np.random.randn(n_dim)

    def f(x):
        return 0.5 * x.T @ Q @ x + b.T @ x

    def grad_f(x):
        return Q @ x + b

    def hess_f(x):
        return Q

    return f, grad_f, hess_f


def benchmark_optimizers(
    dimensions: List[int], n_trials: int = 5, max_iter: int = 100
) -> Dict[str, Dict[int, List[float]]]:
    """
    Benchmark the three optimizers across different dimensions.

    Args:
        dimensions: List of dimensions to test
        n_trials: Number of trials per dimension
        max_iter: Maximum iterations for optimizers

    Returns:
        Dictionary containing runtime data for each optimizer
    """
    results = {
        "Newton": {dim: [] for dim in dimensions},
        "BFGS": {dim: [] for dim in dimensions},
        "L-BFGS": {dim: [] for dim in dimensions},
    }

    for dim in dimensions:
        print(f"Testing dimension: {dim}")

        for trial in range(n_trials):
            # Create problem
            f, grad_f, hess_f = create_quadratic_problem(dim)
            x0 = np.random.randn(dim)

            # Newton
            optimizer = newton.Newton(f, grad_f, hess_f, max_iter=max_iter)
            start_time = time.time()
            optimizer.optimize(x0, return_history=False)
            newton_time = time.time() - start_time
            results["Newton"][dim].append(newton_time)

            # BFGS
            optimizer = bfgs.BFGS(f, grad_f, max_iter=max_iter)
            start_time = time.time()
            optimizer.optimize(x0, return_history=False)
            bfgs_time = time.time() - start_time
            results["BFGS"][dim].append(bfgs_time)

            # L-BFGS
            optimizer = lbfgs.LBFGS(f, grad_f, max_iter=max_iter)
            start_time = time.time()
            optimizer.optimize(x0, return_history=False)
            lbfgs_time = time.time() - start_time
            results["L-BFGS"][dim].append(lbfgs_time)

    return results


def plot_runtime_comparison(
    results: Dict[str, Dict[int, List[float]]], save_path: str = None
):
    """
    Plot runtime comparison with error bars.

    Args:
        results: Runtime results from benchmark_optimizers
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 8))

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Extract data
    dimensions = sorted(next(iter(results.values())).keys())

    # Plot with error bars
    for algorithm, data in results.items():
        means = [np.mean(data[dim]) for dim in dimensions]
        stds = [np.std(data[dim]) for dim in dimensions]

        plt.errorbar(
            dimensions,
            means,
            yerr=stds,
            label=algorithm,
            marker="o",
            capsize=5,
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )

    plt.xlabel("Dimension", fontsize=14)
    plt.ylabel("Runtime (seconds)", fontsize=14)
    plt.title("Runtime Comparison of Optimization Algorithms", fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Set logarithmic scale if the range is large
    if max(dimensions) / min(dimensions) > 10:
        plt.xscale("log")
        plt.yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_runtime_scaling(
    results: Dict[str, Dict[int, List[float]]], save_path: str = None
):
    """
    Plot runtime scaling with theoretical complexity lines.

    Args:
        results: Runtime results from benchmark_optimizers
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 8))

    dimensions = sorted(next(iter(results.values())).keys())

    # Plot actual runtimes
    for algorithm, data in results.items():
        means = [np.mean(data[dim]) for dim in dimensions]
        plt.plot(dimensions, means, "o-", label=algorithm, linewidth=2, markersize=8)

    # Add theoretical complexity lines
    dim_array = np.array(dimensions)

    # Newton: O(n^3) for matrix inversion
    newton_theoretical = (
        dim_array**3 / (dimensions[0] ** 3) * np.mean(results["Newton"][dimensions[0]])
    )
    plt.plot(dimensions, newton_theoretical, "--", label="O(n³) theoretical", alpha=0.7)

    # BFGS: O(n^2) for matrix updates
    bfgs_theoretical = (
        dim_array**2 / (dimensions[0] ** 2) * np.mean(results["BFGS"][dimensions[0]])
    )
    plt.plot(dimensions, bfgs_theoretical, "--", label="O(n²) theoretical", alpha=0.7)

    # L-BFGS: O(n) for two-loop recursion
    lbfgs_theoretical = (
        dim_array / dimensions[0] * np.mean(results["L-BFGS"][dimensions[0]])
    )
    plt.plot(dimensions, lbfgs_theoretical, "--", label="O(n) theoretical", alpha=0.7)

    plt.xlabel("Dimension", fontsize=14)
    plt.ylabel("Runtime (seconds)", fontsize=14)
    plt.title("Runtime Scaling Analysis", fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":

    def rosenbrock(x, a=1.0, b=10.0):
        """Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2"""
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    def rosenbrock_gradient(x, a=1.0, b=10.0):
        """Gradient of the Rosenbrock function"""
        dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
        dy = 2 * b * (x[1] - x[0] ** 2)
        return np.array([dx, dy])

    import bfgs as bfgs
    import lbfgs as lbfgs
    import newton as newton

    initial_point = np.array([-1.5, 2.0])

    bfgs_optimizer = bfgs.BFGS(rosenbrock, rosenbrock_gradient, max_iter=100)
    #     bfgs_optimal_point, bfgs_optimal_value, bfgs_history_x, bfgs_history_f = bfgs_optimizer.optimize(
    #         initial_point
    #     )
    #     bfgs_fig, bfgs_ax = plot_contour_2d(
    #         bfgs_history_x, rosenbrock, "BFGS",save=True
    #     )

    lbfgs_optimizer = lbfgs.LBFGS(rosenbrock, rosenbrock_gradient, max_iter=100)
    #     lbfgs_optimal_point, lbfgs_optimal_value, lbfgs_history_x, lbfgs_history_f = lbfgs_optimizer.optimize(
    #         initial_point
    #     )
    #     lbfgs_fig, lbfgs_ax = plot_contour_2d(
    #         lbfgs_history_x,
    #         rosenbrock,
    #         "L-BFGS",
    #         save=True
    #     )

    newton_optimizer = newton.Newton(rosenbrock, rosenbrock_gradient, max_iter=100)
    #     newton_optimal_point, newton_optimal_value, newton_history_x, newton_history_f = newton_optimizer.optimize(
    #         initial_point
    #     )
    #     newton_fig, newton_ax = plot_contour_2d(
    #         newton_history_x, rosenbrock, "Newton",save=True
    #     )

    #     bfgs_result = bfgs_optimizer.optimize(initial_point)
    #     newton_result = newton_optimizer.optimize(initial_point)
    #     lbfgs_result = lbfgs_optimizer.optimize(initial_point)

    #     results = {
    #     'BFGS': (bfgs_result),
    #     'Newton': newton_result,
    #     'L-BFGS': lbfgs_result
    # }

    #     plot_loss_comparison(
    #     results,
    #     title="Optimization Algorithm Comparison",
    #     log_scale=True,
    #     show_convergence_rate=True,
    #     save_path="./figures/optimization_comparison.png",
    # )
    dimensions = [10, 100, 1000, 10000]

    # Run benchmark
    print("Starting benchmark...")
    results = benchmark_optimizers(dimensions, n_trials=5, max_iter=100)

    # Create plots
    plot_runtime_comparison(results, "runtime_comparison.png")
    plot_runtime_scaling(results, "runtime_scaling.png")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"{'Algorithm':<10} | {'Dim':<5} | {'Mean Time (s)':<15} | {'Std Dev':<10}")
    print("-" * 50)

    for algorithm in results:
        for dim in dimensions:
            times = results[algorithm][dim]
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(
                f"{algorithm:<10} | {dim:<5} | {mean_time:<15.6f} | {std_time:<10.6f}"
            )
        print("-" * 50)

    # Run the comparison
