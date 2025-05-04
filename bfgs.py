import warnings
from typing import Callable, List, Tuple

import numpy as np


class BFGS:
    """BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimizer."""
    
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
        eps: float = 1e-8,  # Small constant for numerical stability
    ):
        self.f = f
        self.grad_f = grad_f
        self.max_iter = max_iter
        self.tol = tol
        self.line_search_max_iter = line_search_max_iter
        self.alpha = alpha
        self.c_armijo = c_armijo
        self.c_wolfe = c_wolfe
        self.eps = eps
    
    def _line_search(
        self,
        x: np.ndarray,
        p: np.ndarray,
        f_x: float,
        grad_f_x: np.ndarray
    ) -> float:
        """Line search with Armijo and Wolfe conditions."""
        alpha = self.alpha
        grad_dot_p = np.dot(grad_f_x, p)
        
        for _ in range(self.line_search_max_iter):
            x_new = x + alpha * p
            f_new = self.f(x_new)
            
            # Armijo condition (sufficient decrease)
            if f_new > f_x + self.c_armijo * alpha * grad_dot_p:
                alpha *= 0.5
                continue
            
            # Wolfe condition (curvature)
            grad_new = self.grad_f(x_new)
            if np.abs(np.dot(grad_new, p)) > self.c_wolfe * np.abs(grad_dot_p):
                alpha *= 0.5
                continue
            
            return alpha
        
        # Fallback: Return last alpha and warn

        warnings.warn("Line search failed to converge.")
        return alpha
    
    def optimize(
        self,
        x0: np.ndarray,
        return_history: bool = True
    ) -> Tuple[np.ndarray, float, List[np.ndarray], List[float]]:
        """Run BFGS optimization."""
        n_dim = len(x0)
        x = x0.copy()
        H_inv = np.eye(n_dim)  # Inverse Hessian approximation
        
        history_x = [x.copy()]
        history_f = [self.f(x)]
        
        for _ in range(self.max_iter):
            grad = self.grad_f(x)
            
            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                break
            
            # Search direction
            p = -H_inv @ grad
            
            # Line search
            alpha = self._line_search(x, p, self.f(x), grad)
            s = alpha * p
            x_new = x + s
            
            # Update inverse Hessian
            grad_new = self.grad_f(x_new)
            y = grad_new - grad
            rho = 1.0 / (np.dot(y, s) + self.eps)
            
            if rho > 0 and np.linalg.norm(s) > 1e-10 and np.linalg.norm(y) > 1e-10:
                V = np.eye(n_dim) - rho * np.outer(s, y)
                H_inv = V @ H_inv @ V.T + rho * np.outer(s, s)
            
            x = x_new
            
            if return_history:
                history_x.append(x.copy())
                history_f.append(self.f(x))
        
        return x, self.f(x), history_x, history_f