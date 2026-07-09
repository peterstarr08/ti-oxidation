import subprocess
import numpy as np

from scipy.optimize import minimize


def lbfgsb_optimize(objective, *args):
    guess = []
    bounds = []
    for param, (lb, ub) in args:
        print(f"Initial guess: {param}\tLower bound: {lb}\tUpper bound: {ub}")
        guess.append(param)
        bounds.append((lb, ub))

    print("Starting the optimization...")
    results = minimize(objective, guess, method='L-BFGS-B', bounds=bounds) 

    print(f"\n\nOpimized param = {results.x}")
    print(f"\n\nOpimized energy = {results.fun} eV")

    return [results.x, results.fun]



    


