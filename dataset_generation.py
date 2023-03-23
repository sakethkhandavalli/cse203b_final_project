import numpy as np
import random

def generate_random_MIP(num_vars, num_constraints):
    # random objective coefficients
    c = np.random.randint(0, 500, num_vars)

    # generate random coefficients for the constraints
    A = np.random.randint(0, 250, size=(num_constraints, num_vars))
    b = np.random.randint(0, 400, num_constraints)

    # fixed bounds
    bounds = [(0, None)] * num_vars

    # randomly assign integer constraints
    int_const = np.random.randint(0, 2, num_vars)

    Aeq = None
    beq = None
    return c, A, b, bounds, int_const, Aeq, beq

def generate_dataset(no_problems):
    problems = []
    for i in range(no_problems):
        num_vars = random.randint(2000, 5000)
        num_constraints = random.randint(1000, 5000)
        problems.append(generate_random_MIP(num_vars, num_constraints))
    return problems