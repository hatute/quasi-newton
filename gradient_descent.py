from typing import Callable, List, Tuple

import numpy as np


class GradientDescent:
    """Gradient Descent optimizer implementation from scratch."""
    
    def __init__(
        self,
        f: Callable,
        grad_f: Callable,
        max_iter: int = 1000,
        tol: float = 1e-6,
        learning_rate: float = 0.01,
        line_search: bool = True,
    ):
        """
        Initialize Gradient Descent optimizer.
        
        Args:
            f: Objective function
            grad_f: Gradient of objective function
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            learning_rate: Initial learning rate
            line_search: Whether to use line search for adaptive step size
        """
        self.f = f
        self.grad_f = grad_f
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.line_search = line_search
    
    def _backtracking_line_search(
        self,
        x: np.ndarray,
        p: np.ndarray,
        alpha_init: float = 1.0,
        rho: float = 0.5,
        c: float = 1e-4,
        max_iter: int = 50
    ) -> float:
        """
        Perform backtracking line search to find appropriate step size.
        
        Args:
            x: Current point
            p: Search direction (negative gradient)
            alpha_init: Initial step size
            rho: Backtracking factor
            c: Armijo condition parameter
            max_iter: Maximum iterations for line search
            
        Returns:
            Optimal step size
        """
        alpha = alpha_init
        f_x = self.f(x)
        grad_x = self.grad_f(x)
        
        for _ in range(max_iter):
            if self.f(x + alpha * p) <= f_x + c * alpha * np.dot(grad_x, p):
                return alpha
            alpha *= rho
        
        return alpha
    
    def optimize(
        self,
        x0: np.ndarray,
        return_history: bool = True
    ) -> Tuple[np.ndarray, float, List[np.ndarray], List[float]]:
        """
        Optimize using gradient descent.
        
        The method iteratively updates the current point:
        x_{k+1} = x_k - alpha_k * grad_f(x_k)
        where alpha_k is the step size (learning rate).
        
        Args:
            x0: Initial point
            return_history: Whether to return optimization history
            
        Returns:
            Optimal point, optimal value, history of points, history of values
        """
        x = x0.copy()
        history_x = [x.copy()]
        history_f = [self.f(x)]
        
        for i in range(self.max_iter):
            # Compute gradient
            grad = self.grad_f(x)
            
            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break
            
            # Compute search direction (negative gradient)
            p = -grad
            
            # Determine step size
            if self.line_search:
                alpha = self._backtracking_line_search(x, p, self.learning_rate)
            else:
                alpha = self.learning_rate
            
            # Update x
            x = x + alpha * p
            
            if return_history:
                history_x.append(x.copy())
                history_f.append(self.f(x))
        
        return x, self.f(x), history_x, history_f