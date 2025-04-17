# CG vs. Quasi-Newton Methods Demo
# Written by Joshua Sheldon
# On April 14, 2025
# Adapted from the following code from the same author:
# https://github.com/LumaDevelopment/AdvancedTopicsInAI/blob/main/ConjugateGradient/polak_ribiere.py

# ---------- IMPORTS ----------

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import line_search, minimize, minimize_scalar, OptimizeResult

# ---------- CLASSES ----------

class Function(ABC):
    """Abstract base class representing a mathematical function."""

    def eval(self, x: np.ndarray):
        """Evaluate the function at x"""
        x = np.asarray(x).flatten()
        return self._eval(x)

    def gradient(self, x: np.ndarray):
        """Calculate the gradient of the function at x"""
        x = np.asarray(x).flatten()
        return self._gradient(x)

    @abstractmethod
    def _eval(self, x: np.ndarray):
        """Evaluation logic, to be implemented by subclasses"""
        pass

    @abstractmethod
    def _gradient(self, x: np.ndarray):
        """Gradient calculation logic, to be implemented by subclasses"""
        pass

class QuadraticForm(Function):
    """A class that represents quadratic form functions."""
    def __init__(self, A: np.ndarray, b: np.ndarray, c: int):
        """
        Arguments:
            A (numpy.ndarray): A matrix.
            b (numpy.ndarray): A vector.
            c (int): A scalar.
        """
        self.A = A
        self.b = np.asarray(b).flatten()
        self.c = c
        pass

    def _eval(self, x: np.ndarray):
        """Evaluate the function at x"""
        return self.c - np.dot(self.b, x) + 0.5 * np.dot(x, np.dot(self.A, x))
    
    def _gradient(self, x: np.ndarray):
        """Calculate the gradient of the function at x"""
        return np.dot(self.A, x) - self.b

class CustomFunction(Function):
    """A class that allows a user to provide a custom function for minimization."""
    def __init__(self, eval_func: callable, grad_func: callable):
        """
        Arguments:
            eval_func (callable): Callable that evaluates f at a point x.
            grad_func (callable): Callable that computes the gradient of f at a point x.
        """
        self.eval_func = eval_func
        self.grad_func = grad_func
    
    def _eval(self, x: np.ndarray):
        """Evaluate the function at x"""
        return self.eval_func(x)
    
    def _gradient(self, x: np.ndarray):
        """Calculate the gradient of the function at x"""
        return self.grad_func(x)

class PolakRibiere:
    """Performs the Polak-Ribiere nonlinear conjugate gradient method."""

    # Constants
    FTOL = 3.0e-8 # Failure to decrease function value by this amount indicates doneness
    ITMAX = 200   # Maximum number of iterations
    EPS = 1.0e-18 # small number to rectify the special case of converging to exactly zero function value
    GTOL = 1.0e-8 # the convergence criterion for the zero gradient test

    def __init__(self, f: Function, P_of_0: np.ndarray):
        """
        Arguments:
            f (Function): The function to minimize.
            P_of_0 (np.ndarray): The starting point.
        """
        self.f = f

        # State tracking variables
        self.i = 0
        self.P = np.asarray(P_of_0.copy()).flatten()
        self.h = -f.gradient(self.P)
        self.g = self.h.copy()
        self.lmbda = None
        self.gamma = None
        self.finished = False

    def iter(self) -> bool:
        """Runs an iteration of Polak-Ribiere.
        
        Returns:
            bool: Whether the algorithm is finished."""
        
        # If the algorithm is done, don't run an iteration
        if self.finished:
            return self.finished
        
        # Step 1: Find lambda that minimizes f(P of i + (Lambda of i * h of i))
        self.lmbda = line_search(self.f.eval, self.f.gradient, self.P, self.h)[0]

        if self.lmbda is None:
            # Wolfe line search failed, resort to Brent
            line = lambda lmbda: self.f.eval(self.P + lmbda * self.h)
            res = minimize_scalar(line)
            if res.success:
                self.lmbda = res.x
            else:
                print(f"Brent line search failed on iteration {self.i + 1}: {res.message}")
                grad_norm = np.linalg.norm(self.g)
                self.lmbda = 0.1 / grad_norm if grad_norm > 1e-8 else 1e-3
                print(f"Fallback λ = {self.lmbda}")
        
        # Step 2: Calculate new point
        old_P = self.P.copy()
        self.P = self.P + self.lmbda * self.h

        # Evaluate f at the old and new point
        old_f = self.f.eval(old_P)
        new_f = self.f.eval(self.P)

        # Check for convergence by function value,
        # set algorithm as finished if converged
        if (2.0 * abs(new_f - old_f) <= 
            PolakRibiere.FTOL * (abs(new_f) + abs(old_f) + PolakRibiere.EPS)):
            self.finished = True
            print(f"Polak-Ribiere converged after {self.i + 1} iterations!")

        # Step 3: Calculate new gradient
        old_g = self.g.copy()
        self.g = -self.f.gradient(self.P)

        # Check for convergence on zero gradient,
        # set algorithm as finished if converged
        test = np.zeros(1)
        den = np.maximum(new_f, 1.0)
        for j in range(len(self.P)):
            temp = abs(self.g[j]) * np.maximum(abs(self.P[j]), 1.0) / den
            if temp > test: test = temp
        if test < PolakRibiere.GTOL:
            self.finished = True
            print(f"Polak-Ribiere converged after {self.i + 1} iterations!")

        # Check for convergence edge case
        gg = np.zeros(1)
        for j in range(len(old_g)):
            gg += old_g[j]*old_g[j]
        if gg == 0.0:
            self.finished = True
            print(f"Polak-Ribiere converged after {self.i + 1} iterations!")

        # Step 4: Calculate gamma
        self.gamma = (np.dot(self.g - old_g, self.g) / 
                      np.dot(old_g, old_g))

        # Step 5: Calculate new search direction
        self.h = self.g + self.gamma * self.h

        # Increment the iteration counter
        self.i += 1

        # If we've hit our max number of iterations, stop
        if self.i >= PolakRibiere.ITMAX:
            self.finished = True
            print(f"Polak-Ribiere has hit maximum iterations! ({self.i})")
        
        # Return whether the algorithm is done
        return self.finished

# ---------- EXAMPLES ----------

# Example 1
ex_1_f = QuadraticForm(np.array([[3, 2], [2, 6]]), np.array([2, -8]), 0)
ex_1_sp = np.array([3, 5])

# Example 2
ex_2_f = CustomFunction(
    lambda x: np.sin(x[0]**2 + x[1]**2) - np.cos(x[0] + x[1]),
    lambda x: np.array([
        2*x[0]*np.cos(x[0]**2 + x[1]**2) + np.sin(x[0] + x[1]),
        2*x[1]*np.cos(x[0]**2 + x[1]**2) + np.sin(x[0] + x[1])
    ])
)
ex_2_sp = np.array([-1, -1])

# Example 3 - Rosenbrock
a = 1
b = 100
ex_3_f = CustomFunction(
    lambda x: (a - x[0])**2 + b*((x[1] - x[0]**2)**2),
    lambda x: np.array([
        (-2)*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2),
        2*b*(x[1] - x[0]**2)
    ])
)
ex_3_sp = np.array([-5, -5])

# ---------- SELECT FUNCTION AND STARTING POINT ----------

func: Function = ex_3_f
sp: np.ndarray = ex_3_sp

# ---------- QUASI-NEWTON ----------

# Stores the path of the optimizer along the function.
qn_path_x = []
qn_path_y = []
qn_path_z = []

qn_path_x.append(sp[0])
qn_path_y.append(sp[1])
qn_path_z.append(func.eval(sp))

# Called for each iteration of BFGS.
# Stores the current point for visualization.
def callback(intermediate_result: OptimizeResult):
    qn_path_x.append(intermediate_result.x[0])
    qn_path_y.append(intermediate_result.x[1])
    qn_path_z.append(intermediate_result.fun)

# Perform minimization
final_or = minimize(
    func.eval, 
    sp, 
    method='BFGS', 
    jac=func.gradient, 
    callback=callback
)

qn_path_x.append(final_or.x[0])
qn_path_y.append(final_or.x[1])
qn_path_z.append(final_or.fun)

print(final_or)

# ---------- POLAK-RIBIERE ----------

# Initialize minimization
pr = PolakRibiere(func, sp)

# Start assembling list of optimization path
# points to graph
path_x = [pr.P[0]]
path_y = [pr.P[1]]
path_z = [pr.f._eval(pr.P)]

# Run steps of Polak-Ribiere and add the 
# resulting points to the list to graph
while not pr.finished:
    # Run algorithm
    pr.iter()

    # Add new point
    path_x.append(pr.P[0])
    path_y.append(pr.P[1])
    path_z.append(pr.f._eval(pr.P))

print(f"Final point of Polak-Ribiere: ({pr.P}, {pr.f._eval(pr.P)})")

# ---------- PLOTTING ----------

# Create a mesh of the function within 
# a range on the x and y axes based on 
# the min/max x/y explored by the 
# minimizers
x_min = min(min(path_x), min(qn_path_x)) - 2
x_max = max(max(path_x), max(qn_path_x)) + 2
y_min = min(min(path_y), min(qn_path_y)) - 2
y_max = max(max(path_y), max(qn_path_y)) + 2

x = np.linspace(x_min, x_max, 400)
y = np.linspace(y_min, y_max, 400)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = pr.f._eval(np.array([X[i, j], Y[i, j]]))

# Create the figure and plot the mesh
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plot the Polak-Ribiere optimization path
ax.plot(path_x, path_y, path_z, 'ro-', linewidth=3, markersize=8, alpha=1.0, label='Conjugate Gradient')

# Plot the quasi-Newton methods optimization path
ax.plot(qn_path_x, qn_path_y, qn_path_z, 'bo-', linewidth=3, markersize=8, alpha=1.0, label='Quasi-Newton')

plt.legend()
plt.show()
