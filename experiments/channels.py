"""
Example (from Nielsen&Chuang) of freedom of choice in the representation of a quantum channel in terms of Kraus operators.
"""
from functools import reduce
from utils import qubit
from sympy import eye, sqrt, symbols, Matrix
from sympy.physics.matrices import msigma

# Arbitrary qubit
r, theta, phi = symbols('r theta phi', real=True, nonnegative=True)
rho = qubit(theta, phi, r)

# Pauli matrices
sx, sy, sz = [msigma(i) for i in (1,2,3)]

# Channel as classical superposition (p=1/2) of trivial map and pi-rotation around Z axis
E = eye(2)/sqrt(2), sz/sqrt(2)

# Channel as projective measurement in the operational basis
F = Matrix([[1,0],[0,0]]), Matrix([[0,0],[0,1]])

def channel(E):
    """Quantum channel in terms of Kraus operators."""
    _sum = lambda iterable: reduce(lambda x, y: x+y, iterable) 
    assert _sum([e*e.conjugate().T for e in E]) == eye(E[0].shape[0])
    return lambda rho: _sum([e*rho*e.conjugate().T for e in E])
    
assert channel(F)(rho) == channel(E)(rho)