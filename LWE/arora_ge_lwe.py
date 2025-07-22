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

# ____________________________________________________ Arora-Ge methods

def LWE_oracle(u, field, d, m = 1, alpha = 1):
    '''Parameters
            u : integer vector with entries from field^n
            field : finite field (of size q)
            d : positive integer
            required 2*d + 1 < q
            alpha : positive real number

        Output
            b : b = A @ u + eta, where A is a random m x n matrix with 
                entries from the field, and "eta" is the noise vector 
                such that each entry is from discrete normal(0, alpha*q)
                conditioned to be of magnitude at most d
    '''

    n = u.shape[0]
    q = field.order()
    A = np.random.choice(field, size = (m, n))
    eta = Discrete_gaussian(0, alpha*q, m)

    # condition ensuring absolute value of eta bounded of d
    for i in range(m):
        if abs(eta[i]) > d:
            eta[i] = sign(eta[i])*d 

    b = A @ u + eta
    return A, b


def AG_linearize(A, b, P, basis):
    '''Arora-Ge linearization algorithm for solving LWE
        Parameters
            A : m, n (numpy) matrix with entries from field
            b : m length (numpy) vector from field
            P : polynomial of degree D
            basis : ORDERED numpy array of all monomials of degree <= D. 
                    |basis| = N = binom(n + D, n)

        Output
            L : numpy vector of length N with entries from the field.
                L is coefficient vector of the polynomial P(A @ z + b) according 
                to the basis "basis"
    '''

    m, n = A.shape # extract parameters

    R_z = basis[0].parent()
    field = basis[0].base_ring()
    z = basis[ :n] # extract variables

    # convert the constraint of the univariate polynomial on "x" to the multivariate polynomial on z1, ..., zn to capture the secret vector u as the solution.
    P_z = [P(i) for i in b - (A @ z)]

    # Linearize the polynomial constraint. The secret vector u is encoded in the kernel of this matrix. 
    L = np.array([Polynomial_to_coefficients(P_z[i], basis) for i in range(m)])
    return L