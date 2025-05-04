import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_contour_2d(
    history_x, f_contour, f_name, x_range=(-2, 2), y_range=(-1, 3), figsize=(12, 8)
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
    levels = np.logspace(0, 2.5, 10)
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

    return fig, ax


def plot_convergence_history(history_f, figsize=(10, 6)):
    """
    Plot convergence history of the optimization.

    Args:
        history_f: List of function values during optimization
        figsize: Figure size as (width, height)

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    iterations = range(len(history_f))
    ax.semilogy(iterations, history_f, "b-", linewidth=2.5, marker="o", markersize=6)

    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Function Value (log scale)", fontsize=14)
    ax.set_title("Convergence History of BFGS on Rosenbrock Function", fontsize=16)
    ax.grid(True, alpha=0.3, which="both")

    # Add annotation for final value
    ax.annotate(
        f"Final value: {history_f[-1]:.6f}",
        xy=(len(history_f) - 1, history_f[-1]),
        xytext=(len(history_f) - 5, history_f[-1] * 10),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=12,
    )

    plt.tight_layout()

    return fig, ax


def plot_2d_contour(
    f: Callable,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    optimizers: Dict[str, List[np.ndarray]],
    title: str = "Optimization Paths on 2D Contour",
    n_points: int = 100,
    ax=None,
):
    """
    Plot 2D contour with optimization paths.

    Args:
        f: Objective function
        x_range: Range for x-axis
        y_range: Range for y-axis
        optimizers: Dictionary of optimizer name to path history
        title: Plot title
        n_points: Number of points for contour
        ax: Optional matplotlib axis to plot on
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Plot contour
    contour = ax.contour(X, Y, Z, levels=20, colors="black", alpha=0.5, linewidths=0.5)
    contourf = ax.contourf(X, Y, Z, levels=20, cmap="gray", alpha=0.7)

    # Add colorbar if creating a new figure
    if ax is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(contourf, cax=cax)

    # Plot optimization paths
    colors = ["green", "red", "blue", "orange", "purple"]
    markers = ["o", "s", "^", "D", "v"]

    for (name, path), color, marker in zip(optimizers.items(), colors, markers):
        path = np.array(path)
        ax.plot(
            path[:, 0], path[:, 1], "-", color=color, alpha=0.7, linewidth=2, label=name
        )
        ax.scatter(path[:, 0], path[:, 1], color=color, marker=marker, s=30)
        # Mark starting point
        ax.scatter(
            path[0, 0],
            path[0, 1],
            color=color,
            marker=marker,
            s=100,
            edgecolor="black",
            linewidth=2,
        )
        # Mark ending point
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            color=color,
            marker=marker,
            s=100,
            edgecolor="white",
            linewidth=2,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_loss_trends(
    optimizers: Dict[str, List[float]],
    title: str = "Loss Trends Comparison",
    log_scale: bool = True,
    ax=None,
):
    """
    Plot loss trends for different optimizers.

    Args:
        optimizers: Dictionary of optimizer name to loss history
        title: Plot title
        log_scale: Whether to use log scale for y-axis
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    colors = ["green", "red", "blue", "orange", "purple"]

    for (name, losses), color in zip(optimizers.items(), colors):
        ax.plot(losses, "-", color=color, linewidth=2, label=name)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss" + (" (log scale)" if log_scale else ""))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_time_vs_loss(
    optimizers: Dict[str, Tuple[List[float], List[float]]],
    title: str = "Running Time vs Loss",
    log_scale: bool = True,
    ax=None,
):
    """
    Plot running time vs loss for different optimizers.

    Args:
        optimizers: Dictionary of optimizer name to (time_history, loss_history)
        title: Plot title
        log_scale: Whether to use log scale for y-axis
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    colors = ["green", "red", "blue", "orange", "purple"]

    for (name, (times, losses)), color in zip(optimizers.items(), colors):
        ax.plot(times, losses, "-", color=color, linewidth=2, label=name)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Loss" + (" (log scale)" if log_scale else ""))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_memory_usage(
    optimizers: Dict[str, List[float]], title: str = "Memory Usage Comparison", ax=None
):
    """
    Plot memory usage for different optimizers.

    Args:
        optimizers: Dictionary of optimizer name to memory usage history
        title: Plot title
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    colors = ["green", "red", "blue", "orange", "purple"]

    for (name, memory), color in zip(optimizers.items(), colors):
        ax.plot(memory, "-", color=color, linewidth=2, label=name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_all_comparisons(
    results: Dict[str, Dict],
    f: Callable,
    x_range: Tuple[float, float] = (-6, 0),
    y_range: Tuple[float, float] = (0, 6),
    save_path: str = None,
):
    """
    Plot all comparisons in a single figure.

    Args:
        results: Dictionary of optimizer results
        f: Objective function for contour plotting
        x_range: Range for x-axis in contour plot
        y_range: Range for y-axis in contour plot
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(16, 20))

    # 1. 2D Contour Plots
    ax1 = fig.add_subplot(4, 1, 1)
    optimizers_paths = {name: res["history_x"] for name, res in results.items()}
    _, _ = plot_2d_contour(f, x_range, y_range, optimizers_paths, n_points=100, ax=ax1)

    # 2. Loss Trends
    ax2 = fig.add_subplot(4, 1, 2)
    optimizers_losses = {name: res["history_f"] for name, res in results.items()}
    _, _ = plot_loss_trends(optimizers_losses, ax=ax2)

    # 3. Time vs Loss
    ax3 = fig.add_subplot(4, 1, 3)
    optimizers_time_loss = {
        name: (res["time_history"], res["history_f"])
        for name, res in results.items()
        if "time_history" in res
    }
    _, _ = plot_time_vs_loss(optimizers_time_loss, ax=ax3)

    # 4. Memory Usage
    ax4 = fig.add_subplot(4, 1, 4)
    optimizers_memory = {
        name: res["memory_usage"]
        for name, res in results.items()
        if "memory_usage" in res
    }
    if optimizers_memory:  # Only plot if there's memory data
        _, _ = plot_memory_usage(optimizers_memory, ax=ax4)
    else:
        ax4.text(
            0.5,
            0.5,
            "No memory usage data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Memory Usage Comparison")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_individual_contours(
    f: Callable,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    optimizers: Dict[str, List[np.ndarray]],
    save_prefix: str = None,
    n_points: int = 100,
):
    """
    Plot individual contour plots for each optimizer.

    Args:
        f: Objective function
        x_range: Range for x-axis
        y_range: Range for y-axis
        optimizers: Dictionary of optimizer name to path history
        save_prefix: Prefix for saving individual plots
        n_points: Number of points for contour
    """
    for name, path in optimizers.items():
        fig, ax = plot_2d_contour(
            f,
            x_range,
            y_range,
            {name: path},
            title=f"{name} Optimization Path",
            n_points=n_points,
        )

        if save_prefix:
            plt.savefig(
                f"{save_prefix}_{name.lower()}.png", dpi=300, bbox_inches="tight"
            )

        plt.show()
        plt.close()


def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2"""
    a = 1.0
    b = 100.0
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def rosenbrock_gradient(x):
    """Gradient of the Rosenbrock function"""
    a = 1.0
    b = 100.0
    dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    dy = 2 * b * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


if __name__ == "__main__":
    import bfgs as bfgs
    import lbfgs as lbfgs
    initial_point = np.array([-1.5, 2.0])

    # optimizer = bfgs.BFGS(rosenbrock, rosenbrock_gradient, max_iter=100)
    # optimal_point, optimal_value, history_x, history_f = optimizer.optimize(
    #     initial_point
    # )
    # fig, ax = plot_contour_2d(
    #     history_x, rosenbrock, "BFGS", x_range=(-2, 2), y_range=(-1, 3), figsize=(12, 8)
    # )

    optimizer = lbfgs.LBFGS(rosenbrock, rosenbrock_gradient, max_iter=100)
    optimal_point, optimal_value, history_x, history_f = optimizer.optimize(
        initial_point
    )
    fig, ax = plot_contour_2d(
        history_x, rosenbrock, "L-BFGS", x_range=(-2, 2), y_range=(-1, 3), figsize=(12, 8)
    )
    plt.show()
