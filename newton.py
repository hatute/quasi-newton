from typing import Callable, List, Tuple

import numpy as np


class Newton:
    """Newton's method optimizer implementation from scratch."""
    
    def __init__(
        self,
        f: Callable,
        grad_f: Callable,
        hess_f: Callable,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        """
        Initialize Newton optimizer.
        
        Args:
            f: Objective function
            grad_f: Gradient of objective function
            hess_f: Hessian of objective function
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
        """
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.max_iter = max_iter
        self.tol = tol
    
    def optimize(
        self,
        x0: np.ndarray,
        return_history: bool = True
    ) -> Tuple[np.ndarray, float, List[np.ndarray], List[float]]:
        """
        Optimize using Newton's method.
        
        The method iteratively updates the current point by solving:
        x_{k+1} = x_k - H_k^{-1} * grad_f(x_k)
        where H_k is the Hessian matrix at x_k.
        
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
            # Compute gradient and Hessian
            grad = self.grad_f(x)
            hess = self.hess_f(x)
            
            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break
            
            # Solve Newton system: H * p = -grad
            try:
                p = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # If Hessian is singular, use gradient descent direction
                print(f"Warning: Singular Hessian at iteration {i}, using gradient descent step")
                p = -grad
            
            # Update x
            x = x + p
            
            if return_history:
                history_x.append(x.copy())
                history_f.append(self.f(x))
        
        return x, self.f(x), history_x, history_f