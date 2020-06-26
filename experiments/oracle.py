"""
I want to show that an quantum oracle for the Kronecker delta on n source qubits plus one target

    U_x0 = (Id_n - 2*|x0><x0|) otimes Id_2

can be implemented with single qubit NOTs on zero bits and a Toffoli gate T_n with source qubits as controls, i.e.

    U_x0 = (otimes_i X^(b_i+1) otimes 1_2) T_n (otimes_i X^(b_i+1) otimes 1_2)

where b_i is the binary expansion of x_0.
"""

from sympy import Matrix, eye
from sympy.physics.quantum import TensorProduct

X = Matrix([[0,1],[1,0]])

def T(n):
    """Toffoli gate with n-1 controls and one target."""

    # TODO
    # m = eye(2**n) - [0, .., 0, 1, 1].T * [0, ..., 0, 1, 1] + [0, sigma_x]
