from typing import Callable, List, Optional, Tuple

import numpy as np


class Newton:
    """
    Newton's Method optimizer with line search and fallback for non-positive-definite Hessians.
    
    Args:
        f: Objective function.
        grad_f: Gradient of the objective function.
        hess_f: Hessian of the objective function (if None, use finite differences).
        max_iter: Maximum iterations.
        tol: Tolerance for termination.
        line_search_max_iter: Maximum line search iterations.
        c_armijo: Armijo condition parameter.
        c_wolfe: Wolfe condition parameter.
        eps: Small constant for numerical stability.
    """
    def __init__(
        self,
        f: Callable,
        grad_f: Callable,
        hess_f: Optional[Callable] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        line_search_max_iter: int = 20,
        c_armijo: float = 1e-4,
        c_wolfe: float = 0.9,
        eps: float = 1e-8,
    ):
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f if hess_f else self._numerical_hessian
        self.max_iter = max_iter
        self.tol = tol
        self.line_search_max_iter = line_search_max_iter
        self.c_armijo = c_armijo
        self.c_wolfe = c_wolfe
        self.eps = eps

    def _numerical_hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian using finite differences (fallback if not provided)."""
        n = len(x)
        hess = np.zeros((n, n))
        grad_0 = self.grad_f(x)
        h = 1e-5  # Step size
        
        for i in range(n):
            x_perturbed = x.copy()
            x_perturbed[i] += h
            grad_perturbed = self.grad_f(x_perturbed)
            hess[:, i] = (grad_perturbed - grad_0) / h
        
        return (hess + hess.T) / 2  # Ensure symmetry

    def _line_search(
        self,
        x: np.ndarray,
        p: np.ndarray,
        f_x: float,
        grad_f_x: np.ndarray,
    ) -> float:
        """Backtracking line search with Armijo-Wolfe conditions."""
        alpha = 1.0
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
        
        return alpha  # Fallback if line search fails

    def optimize(
        self,
        x0: np.ndarray,
        return_history: bool = True,
    ) -> Tuple[np.ndarray, float, List[np.ndarray], List[float]]:
        """
        Run Newton's method optimization.
        
        Returns:
            x_opt: Optimal point.
            f_opt: Optimal function value.
            x_hist: History of points (if return_history=True).
            f_hist: History of function values (if return_history=True).
        """
        x = x0.copy()
        history_x = [x.copy()]
        history_f = [self.f(x)]
        
        for _ in range(self.max_iter):
            grad = self.grad_f(x)
            hess = self.hess_f(x)
            
            if np.linalg.norm(grad) < self.tol:
                break
            
            try:
                # Solve Newton direction: p = -H^{-1} ∇f
                p = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent if Hessian is singular
                p = -grad
            
            # Line search
            alpha = self._line_search(x, p, self.f(x), grad)
            x_new = x + alpha * p
            
            if return_history:
                history_x.append(x_new.copy())
                history_f.append(self.f(x_new))
            
            x = x_new
        
        return x, self.f(x), history_x, history_f