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

# ______________________________________________ solver using Grobner basis


def Grobner_basis_solver(A, b, P):
    '''Parameters
            A : numpy matrix m x n, entries from finite field
            b : numpy vector of length m, entry from finite field
            P : univariate polynomial, coefficients from finite field

        Output
            u
            u is numpy vector, entries from the finite field, solution to the 
            LWE instance posed by the pair (A, b). That is, b = A @ u + eta
            for some eta generated from discrete gaussian with entry bound of d. 
    '''

    # convert the constraint of the univariate polynomial on "x" to the multivariate polynomial on z1, ..., zn to capture the secret vector u as the solution.
    q = P.base_ring().order()
    n = A.shape[-1]
    R_z = PolynomialRing(GF(q), n, names = 'z') # parent ring
    z = np.array(R_z.gens())
    
    P_z = [P(i) for i in b - (A @ z)]

    u = []
    for p in R_z.ideal(P_z).groebner_basis(): # grobner basis solver
        u.append(-p.constant_coefficient())

    return np.array(u)


