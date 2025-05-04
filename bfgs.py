from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class BFGS:
    """BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimizer implementation."""
    
    def __init__(
        self,
        f: Callable,
        grad_f: Callable,
        max_iter: int = 500,
        tol: float = 1e-6,
        line_search_max_iter: int = 20,
        alpha: float = 1.0,
        c_armijo: float = 1e-4,
        c_wolfe: float = 0.9,
    ):
        """
        Initialize BFGS optimizer.
        
        Args:
            f: Objective function
            grad_f: Gradient of objective function
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            line_search_max_iter: Maximum iterations for line search
            alpha: Initial step size
            c1: Armijo condition parameter
            c2: Wolfe condition parameter
        """
        self.f = f
        self.grad_f = grad_f
        self.max_iter = max_iter
        self.tol = tol
        self.line_search_max_iter = line_search_max_iter
        self.alpha = alpha
        self.c_armijo = c_armijo
        self.c_wolfe = c_wolfe
    
    def _line_search(
        self,
        x: np.ndarray,
        p: np.ndarray,
        f_x: float,
        grad_f_x: np.ndarray
    ) -> float:
        """
        Perform line search to find optimal step size using Wolfe conditions.
        
        Args:
            x: Current point
            p: Search direction
            f_x: Function value at x
            grad_f_x: Gradient at x
            
        Returns:
            Optimal step size
        """
        alpha = self.alpha
        grad_dot_p = np.dot(grad_f_x, p)
        
        # Backtracking line search
        for _ in range(self.line_search_max_iter):
            x_new = x + alpha * p
            f_new = self.f(x_new)
            
            # Check Armijo condition (sufficient decrease)
            if f_new > f_x + self.c_armijo * alpha * grad_dot_p:
                alpha *= 0.5
                continue
            
            # Wolfe condition (curvature)
            grad_new = self.grad_f(x_new)
            if np.dot(grad_new, p) < self.c_wolfe * grad_dot_p:
                alpha *= 0.5
                continue

            return alpha  # Successful line search
        
        return alpha # Fallback if line search fails

        
        return alpha
    
    def optimize(
        self,
        x0: np.ndarray,
        return_history: bool = True
    ) -> Tuple[np.ndarray, float, List[np.ndarray], List[float]]:
        """
        Optimize using BFGS algorithm.
        
        Args:
            x0: Initial point
            return_history: Whether to return optimization history
            
        Returns:
            Optimal point, optimal value, history of points, history of values
        """
        n_dim = len(x0)
        x = x0.copy()
        H_inv = np.eye(n_dim)  # Initial inverse Hessian approximation (identity matrix)
        
        history_x = [x.copy()]
        history_f = [self.f(x)]
        
        for i in range(self.max_iter):
            f_val = self.f(x)
            grad = self.grad_f(x)
            
            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break
            
            # Compute search direction
            p = -H_inv @ grad
            
            # Line search
            alpha = self._line_search(x, p, f_val, grad)
            
            # Update position
            s = alpha * p
            x_new = x + s
            
            # Compute gradient at new point
            grad_new = self.grad_f(x_new)
            
            # Update inverse Hessian using BFGS formula
            y = grad_new - grad
            rho = 1.0 / np.dot(y, s)
            
            if rho > 0:  # Ensure positive definiteness
                V = np.eye(n_dim) - rho * np.outer(s, y)
                H_inv = V @ H_inv @ V.T + rho * np.outer(s, s)
            
            x = x_new
            
            if return_history:
                history_x.append(x.copy())
                history_f.append(self.f(x))
        
        return x, self.f(x), history_x, history_f
    
