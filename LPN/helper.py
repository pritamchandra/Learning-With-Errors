# ________________________________________________ Library Imports

from sage.all import *
from itertools import product
from scipy.linalg import solve
from scipy.special import binom
from math import prod

import numpy as np
import galois

# ________________________________________________ Helper Functions

def Galois_kernel(L, q = 2):
    '''Return the kernel of the matrix LP over finite field GF(q) 
    using the galois library.
    '''
    GFq = galois.GF(int(q))
    LP_galois = GFq(L.astype(np.int32))
    return LP_galois.null_space()

def Kronecker_power(A, k):
    '''Returns the Kronecker power of A, i.e. A^k
    '''
    if k == 1:
        return A

    else:
        # can be made more efficient with the power method of exponentiation
        return np.kron(A, Kronecker_power(A, k - 1))

def All_monoms(variables, degree, mutilinear_only = True):
    '''Return all monomials less than the given degree d in the variables.
    If mutilinear_only is True, then only the monomials which are
    mutilinear are returned.

    NOTE: this is the most expensive part of the algorithm currently. 
    Can be optimized by using power method in Knronecker_power, or even 
    further by generating all subsets directly, instead of obtaining them
    from product set. 
    '''
    monoms = [] # the constant monomial
    for i in range(1, degree + 1):
        # all monomials of degree i
        monoms += list(dict.fromkeys(Kronecker_power(variables, i)))        

    # Note that dict.fromkeys removes duplicates like "set", but keeps 
    # the order deterministic, unlike "set". Thus All_monoms is 
    # deterministic, which is important as this set serves as a basis.

    if mutilinear_only:
        # check for multilinearity
        monoms = [monom for monom in monoms if monom.degree() == len(monom.variables())]

    return np.array(monoms + [GF(2).one()]) # add the constant monomial

def Random_root(P, q = None, max_trials = 1e3):
    '''Randomly search for a root of the polynomial P in GF(q)^m within 
    max_trials number of trials.

    NOTE: Alternatively one can generate all the roots of P and pick
    randomly from the set; that is expensive but always works. This approach 
    much faster in the average case, but may fail sometimes. 
    '''
    n = len(P.parent().gens()) # number of variables in P
        
    if q is None:
        q = P.parent().base_ring().order() # field order

    trial = 0
    while trial < max_trials:
        # try a random element of GF(q)^n on P
        root = np.random.choice(GF(q), size = n)
        if P(list(root)) == 0:
            return root # root found
        trial += 1
    
    # root not found
    return None 
    
def Generate_polynomial(R, d, max_trials = 1e3):
    '''Generate a random nonconstant polynomial of degree d from the 
    ring R with at least one root. The paramter "max_trials" is fed
    to the Random_root function. 
    '''
    P = R(0); root = None
    while P.degree() < 1 or root is None:
        P = R.random_element(degree = d)
        root = Random_root(P, max_trials = max_trials)  

    return P


