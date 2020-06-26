"""
Partial collapse of an n-qubit entangled state.
"""

import copy
from functools import reduce

from sympy import symbols, refine, Q
from sympy import sqrt, log
from sympy import Matrix, diag


def basis(n):
    """
    Standard basis in C^2n.
    """

    return [Matrix([1 if i==j else 0 for i in range(2**n)]) for j in range(2**n)]


def outer(a,b):
    """
    Outer product of two vectors.
    """
    a = a.T if a.shape[1] != 1 else a
    b = b.T if b.shape[0] != 1 else b

    return a*b


def ghz(n):
    """
    N-qubit GHZ state.
    """
    ket0, ket1 = map(lambda j: basis(n)[j], [0,-1])

    return (ket0+ket1)/sqrt(2)


def refineMatrix(m, assumptions=[]):
    """
    Simplify matrix element by element using assumptions.
    """

    if not hasattr(assumptions, '__len__'):
        assumptions = [assumptions]

    return Matrix([[refine(item, *assumptions) for item in row] for row in m.tolist()])


def projector(n,i,k):
    """
    Projector acting on an n-qubit state, projecting the i-th qubit onto the |k> subspace.

    @param n: Positive integer. Number of qubits.
    @param i: Positive integer. Qubit to be measured.
    @param k: Boolean or {0, 1}. Subspace to project i-th qubit onto.
    """
    return diag(*[item for sub in [[2**k%2]*(2**(n-i-1))+[2**(not k)%2]*(2**(n-i-1))]*(2**i) for item in sub])


def supOp(op):
    """
    Return superoperator function associated to operator.
    """
    return lambda rho: op*rho*op.conjugate().T


def projectQubits(rho, j):
    """
    Projective measurements of a subset of qubits.
    """

    if rho.rows != rho.cols:
        if 1 in rho.shape:
            rho = outer(rho,rho)
        else:
            raise ValueError('Dimension mismatch! rho must be density matrix or ket!')

    n = log(rho.rows, 2)

    if not int(n) == n:
        raise ValueError('Density matrix or ket dimension must be a power of 2!')

    new = copy.deepcopy(rho)

    if not hasattr(j, '__len__'):
        j = [j]

    for i in j:
        new = supOp(projector(n,i,0))(new) + supOp(projector(n,i,1))(new)

    return new


def ensembleToMatrix(ensemble):
    """
    Transform iterable of tuples (probability, ket) to density matrix.
    """

    return reduce(lambda x,y: x+y, [prob*ket*ket.T for prob, ket in ensemble])


if __name__ == '__main__':

    # Dimensionality
    n = 2

    # C^4 basis
    ket00, ket01, ket10, ket11 = basis(n)

    # Probability
    p = symbols('p', positive = True)

    # Initial entangled pure state is a "quantum Bernouilli" with success = |11> and fail = |00>
    psi = sqrt(1-p)*ket00 + sqrt(p)*ket11 # ket
    rho_pure = outer(psi,psi) # density matrix

    # Projective measurements on 1st qubit
    j = 0
    proj0 = projector(n,j,0)
    proj1 = projector(n,j,1)

    # Measurement w/ kets ~ Entangled state -> Mixed state: "classical Bernouilli" w/ Pr[|11>]=p; Pr[|00>]=1-p
    ensemble = ((1-p), (proj0*psi).normalized()), (p, (proj1*psi).normalized())
    rho_mixed_kets = refineMatrix(ensembleToMatrix(ensemble), Q.positive(sqrt(1-p)))

    # Measurement w/ superoperator ~ rho -> S[rho] = sum_i proj_i*rho*proj_i.T
    rho_mixed_super = projectQubits(rho_pure, j)
