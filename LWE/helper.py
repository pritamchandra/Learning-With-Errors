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


# ____________________________________________________ Helper functions

def Discrete_gaussian(u, s, shape):
    ''' Parameters
            u : real number (mean)
            s : positive real number (standard deviation)
            shape : array shape 

        Output
            array with shape "shape" with each entry as the nearest integer to
            independently sampled normal(u, s)
    ''' 
    
    return np.round(np.random.normal(u, s, shape)).astype(int)

def Kronecker_power(A, k):
    ''' Parameters
            A : numpy matrix or vector
            k : positive integer

        Output
            A^k, the k-th kronecker power of A
    '''
    if k == 1:
        return A

    else:
        # can be made more efficient with the power method of exponentiation
        return np.kron(A, Kronecker_power(A, k - 1))


def All_monomials(variables, degree, mutilinear_only = False):
    ''' Parameters
            variables : list of generators of the polynomial
                        like x1, x2, ...
            degree : positive integer
            multilinear_only : boolean

        Output
            basis : (ordered) numpy array of all monomials in the variables
                    "variables" with degree <= d.

                    When multilinear_only = True, return only multilinear
                    monomials. Like x1*x2*x3 and not (x1^2)*x2. This attribute
                    is useful for LPN.
        
        NOTE: this is the most expensive part of the algorithm currently. 
        Can be optimized by using power method in Knronecker_power, or even 
        further by generating all subsets directly, instead of obtaining them
        from product set. 
    '''
    q = variables[0].parent().base_ring().order()

    basis = [] 
    for i in range(1, degree + 1):
        # all monomials of degree i
        basis += list(dict.fromkeys(Kronecker_power(variables, i)))        

    # Note that dict.fromkeys removes duplicates like "set", but keeps 
    # the order deterministic, unlike "set". Thus All_monoms is 
    # deterministic, which is important as this set serves as a basis.
    if mutilinear_only:
        # check for multilinearity: degree = #variables
        basis = [monomial for monomial in basis if monomial.degree() == len(monomial.variables())]

    basis = np.array(basis + [GF(q).one()]) # add the constant monomial
    
    return basis

def Polynomial_to_coefficients(P, basis):
    ''' Parameters
            P : polynomial
            basis : numpy 1D array of monomials 

        Output
            C : numpy 1D array of size |basis| with entries from 
                the base ring of the polynomial. C represents the polynomial
                in a coefficient vector format, according to the basis "basis"
    '''

    
    monomials, coefficients = P.monomials(), P.coefficients()
    # expected that P = sum(coefficients[i] * monomials[i])

    C = np.array([P.base_ring().zero()] * basis.shape[-1])
    for i in range(len(monomials)):
        C[basis == monomials[i]] = coefficients[i]

    return C

def Galois_kernel(L, q = 2):
    '''Return the kernel of the matrix LP over finite field GF(q) 
    using the galois library.
    '''
    GFq = galois.GF(int(q))
    LP_galois = GFq(L.astype(np.int32))
    return LP_galois.null_space()