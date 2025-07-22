# ___________________________________________________ Library imports

from sage.all import *
from itertools import product
from scipy.linalg import solve
from scipy.special import binom
from math import prod

import pandas as pd
import numpy as np
import galois
import matplotlib.pyplot as plt

# global options (necessary for python?)

np.set_printoptions(legacy = '1.25', threshold = sys.maxsize)
pd.set_option('display.max_columns', None)

from helper import *
from arora_ge_lwe import *
from grobner_lwe import *

# MAIN

# # _________________________________ Parameters
q = 13 # all operations are in GF(q)
field = GF(q)
m = 150 # dimension of the error vector eta
n = 4 # dimension of the secret vector u
d = 2 # error bound on eta
D = 2*d + 1 
alpha = 0.1

# # _________________________________

# # REQUIRE alpha * q < sqrt(n) and q >> [alpha * q * log(n)]^2
# N * alpha * q^2 * log(q) required number of queries ... comparable to m

# R.<x> = GF(q)[] # python unable to read this syntax

R = PolynomialRing(GF(q), 'x')
x = R.gens()[0]
# the solution of LWE would be recovered as a solution to a polynomial
# equation involving the polynomial P
P = x * prod([(x + i)*(x - i) for i in range(1, d + 1)])

R_z = PolynomialRing(GF(q), n, names = 'z') 
# the z variables are to capture u as a solution of polynomial equation
z = np.array(R_z.gens()) # variables

# basis of monomials for polynomials in z1, ..., zn of degree <= D
basis = All_monomials(z, D, mutilinear_only = False)
# m = int(len(basis) * alpha * (q**2) * log(q)) # required number of queries?
m = len(basis) + 1

# Algorithm instance
u = np.random.choice(GF(q), size = n) # secret vector
print("Secret vector: ", u)

A, b = LWE_oracle(u, field = field, d = d, m = m, alpha = alpha)
# __________________________________________________________

# Grobner basis method
print("Grobner basis solution: ", Grobner_basis_solver(A, b, P))


# Arora-ge method
L = AG_linearize(A, b, P, basis)
# why does this work when it works? 
# calculate the kernel of L and take the appropriate subset of vectors
sol = np.array(Matrix(L.T).kernel().basis())[ : , n : 2*n]
print("Arora-Ge solution:", sol) 
