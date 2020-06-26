"""
Can quantum teleportation be done without classical communication by using a Grover kernel in Alice's lab?

NO: local unitaries cannot break entanglement. From the Wikipedia page on [separable states](https://en.wikipedia.org/wiki/Separable_state):

    > In terms of quantum channels, a separable state can be created from any other state using local actions and classical communication while an entangled state cannot.
"""

from functools import reduce

from sympy import symbols, eye, sqrt, Matrix, Rational, pprint
from sympy.matrices.dense import matrix_multiply_elementwise as schur
from sympy.physics.quantum import TensorProduct as TP

from grover import ParametricGrover as pg

# 1-qubit computational basis
cb1 = [eye(2).col(i) for i in range(2)]

# 2-qubit computational basis
cb2 = [TP(ei, ej) for ei in cb1 for ej in cb1]

# Bell basis parametrised by second bit in first ket and parity, i.e.
#
#  B(b, p) = |0b> + (-1)^p |1bbar>
#
# with b, p in {0, 1} and bbar = not b
B = lambda b, p: sqrt(Rational(1,2)) * (cb2[b] + (-1)**p * cb2[2+(not b)])
bb = [B(b,p) for b in (0,1) for p in (0,1)]

# Change of basis: computational -> Bell
C = Matrix([b.T for b in bb]).T

# qbit we want to send in the computational basis
a, b = symbols('alpha beta')
phi = a*cb1[0] + b*cb1[1]

# Total state
Psi = lambda k: TP(phi, bb[k])

# Grover kernel for a 2-qubit element k in the Bell basis
# TODO why do I get a minus sign?
K = lambda k: -C*pg(2**2)._K(k)*C.T

# Global unitary
U = lambda k: TP(K(k), eye(2))

# Projector onto Bell state
P = lambda k: TP(bb[k], bb[k].T)

# Global measurement
M = lambda k: TP(P(k), eye(2))

# TODO implement trace over A
