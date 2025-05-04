from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np


class LBFGS:
    """Limited-memory BFGS (L-BFGS) optimizer."""
    
    def __init__(
        self,
        f: Callable,
        grad_f: Callable,
        m: int = 10,  # Memory size
        max_iter: int = 500,
        tol: float = 1e-6,
        line_search_max_iter: int = 20,
        c_armijo: float = 1e-4,
        c_wolfe: float = 0.9,
    ):
        self.f = f
        self.grad_f = grad_f
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.line_search_max_iter = line_search_max_iter
        self.c_armijo = c_armijo
        self.c_wolfe = c_wolfe
        self.s_history = deque(maxlen=m)  # Store past 's' vectors (x differences)
        self.y_history = deque(maxlen=m)  # Store past 'y' vectors (gradient differences)
    
    def _line_search(self, x: np.ndarray, p: np.ndarray, f_x: float, grad_f_x: np.ndarray) -> float:
        alpha = 1.0
        grad_dot_p = np.dot(grad_f_x, p)
        
        for _ in range(self.line_search_max_iter):
            x_new = x + alpha * p
            f_new = self.f(x_new)
            
            # Armijo condition
            if f_new > f_x + self.c_armijo * alpha * grad_dot_p:
                alpha *= 0.5
                continue
            
            # Wolfe condition
            grad_new = self.grad_f(x_new)
            if np.abs(np.dot(grad_new, p)) > self.c_wolfe * np.abs(grad_dot_p):
                alpha *= 0.5
                continue
            
            return alpha
        
        return alpha
    
    def _two_loop_recursion(self, grad: np.ndarray) -> np.ndarray:
        """Compute search direction using past {s, y} pairs."""
        q = grad.copy()
        alphas = []
        
        # First loop (backward)
        for s, y in zip(reversed(self.s_history), reversed(self.y_history)):
            rho = 1.0 / np.dot(y, s)
            alpha = rho * np.dot(s, q)
            alphas.append(alpha)
            q -= alpha * y
        
        # Apply initial Hessian approximation (scaled identity)
        if len(self.y_history) > 0:
            last_y, last_s = self.y_history[-1], self.s_history[-1]
            gamma = np.dot(last_y, last_s) / np.dot(last_y, last_y) if len(self.y_history) > 0 else 1.0  # Scaling factor
            r = gamma * q
        else:
            r = q
        
        # Second loop (forward)
        for s, y, alpha in zip(self.s_history, self.y_history, reversed(alphas)):
            rho = 1.0 / np.dot(y, s)
            beta = rho * np.dot(y, r)
            r += s * (alpha - beta)
        
        return -r
    
    def optimize(
        self,
        x0: np.ndarray,
        return_history: bool = True
    ) -> Tuple[np.ndarray, float, List[np.ndarray], List[float]]:
        x = x0.copy()
        history_x = [x.copy()]
        history_f = [self.f(x)]
        
        for _ in range(self.max_iter):
            grad = self.grad_f(x)
            
            if np.linalg.norm(grad) < self.tol:
                break
        
            # Compute search direction to replace the p = -H_inv @ grad
            p = self._two_loop_recursion(grad)
            
            # Line search
            alpha = self._line_search(x, p, self.f(x), grad)
            
            # Update position
            s = alpha * p
            x_new = x + s
            
            # Store {s, y} pairs
            grad_new = self.grad_f(x_new)
            y = grad_new - grad
            self.s_history.append(s)
            self.y_history.append(y)
            
            x = x_new
            
            if return_history:
                history_x.append(x.copy())
                history_f.append(self.f(x))
        
        return x, self.f(x), history_x, history_f