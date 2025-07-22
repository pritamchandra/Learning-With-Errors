# ________________________________________________ Library Imports

from sage.all import *
from itertools import product
from scipy.linalg import solve
from scipy.special import binom
from math import prod

import numpy as np
import galois

from helper import *
from arora_ge_lpn import *

# _____________________________________________ parameters

q = 2 # all operations are in GF(q)
m = 30 # dimension of the error vector eta
n = 26 # dimension of the secret vector u
d = 2 # degree of the polynomial P, whose roots are the noise
max_trials = 1e3 # threshold for the number of trials for generating polynomial P

# number of non constant monomials in n variables of degree <= d
N = int(sum([binom(int(n), i) for i in range(1, d + 1)]))
recommended_sample_size = 10 * N * (2**(m + d)) # sample size recommended by Arora-Ge
sample_size = 2*N # batch size (# of oracle calls) to linearoize the problem

# _____________________________________________ generation

R_x = PolynomialRing(GF(q), m, names = 'x') # Parent ring of P
u = np.random.choice(GF(q), size = n) # secret vector

print("Generating polynomial P ...", end = " ")
P = Generate_polynomial(R_x, d, max_trials = max_trials)
print(P)

# _______________________________________________ solution

# LP for "Linearization of P"
LP = Linearize_batch(u, P, sample_size = sample_size,
                           randomize_eta = True,
                           print_constraints = False) 

print("Gaussian elimination.")
LP_kernel = Galois_kernel(LP)

# the solutions are extracted as the coefficients corresponding 
# to the monomials z_i
sols = np.unique(LP_kernel[ : , :n], axis = 0)

# ________________________________________________ output

success_flag = False
print("\nAlgorithm solutions:")
for sol in sols: 
    print(sol)
    if (sol == u).all():
        success_flag = True

print("\nSecret vector:", u)
print("\nSuccess!" if success_flag else "Failure!")

