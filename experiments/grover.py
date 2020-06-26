"""
cirq implementation of Grover's algorithm.
"""

import cirq
import numpy as np
import matplotlib.pyplot as plt

from functools import reduce

import sympy
from sympy import Matrix, Rational, I, sqrt, asin, pi, eye
from sympy.physics.quantum import TensorProduct

class Qubit(sympy.Matrix):

    def __init__(self, coords):
        """
        Create qubit.

        @params coords: Coordinates in the {|0>, |1>} basis. Must be normalised.
        """

        a, b = coords

        assert abs(a)**2 + abs(b)**2 == 1, 'Qubit amplitudes must be normalised!'

        #self.a = a
        #self.b = b

        # TODO make this work
        super(Qubit, self).__init__(coords)

#def Ket(sympy.Matrix)

class ParametricGrover:
    # TODO plots with prob and prob v. m as a function of alpha, beta, gamma, delta
    # TODO generalise to amplitude amplification. This involves two things:
    #
    # - Make G1 an arbitrary rank-one projector onto ANY state (that's part of the input now) instead of the uniform ket
    # - Make G2 an arbitrary projector onto ANY subspace of ANY dimension (again that's part of the input) instead of the rank-one projector onto x0
    #
    # The point is: if you give me an initial state psi and a projector P onto a subspace H_1, AA sends psi to H_1 in m = pi/4*theta (exact equality) steps, where theta = arcsin(|P psi|) and each step is the application of a unitary that can be written explicitly in terms of P and psi. In fact, it is just
    #
    # Q = - (I-2|psi><|psi|) (I-2P)
    #
    # for the optimal choice alpha=gamma=-1 and beta=delta=1
    #
    # NB: if psi is the uniform ket wrt to the computational basis and P = |x0><x0| for some x0 in the computational basis, |P psi| = 1/sqrt(N). For big enough N, arcsin(1/sqrt(N)) ~ sqrt(N) and so m ~ O(sqrt(N)) <- this is the origin of the quadratic speedup!
    #
    # Notice how this would be different if quantum states where "classical" i.e. unitary wrt the L1 norm instead of L2. Then psi would be on the simplex and its projections would be of order 1/N, thus spoiling the speedup!

    def __init__(self, N, alpha=-1, beta=1, gamma=-1, delta=1):

        ### CONSTANTS
        # Items in database
        self.N = N

        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Number of qubits in QC
        self.n = int(np.ceil(np.log2(self.N)))

        # Dimension of product space
        self.dim = 2**self.n

        ### GATES
        # Hadamard gate
        self.H = sqrt(Rational(1,2))*Matrix([
            [1, 1],
            [1, -1]
            ])

        # Hadamard layer - one H per qubit
        self.H_layer = reduce(TensorProduct, [self.H for qbit in range(self.n)])

        # Uniform ket
        #self._u = sqrt(Rational(1,self.dim)) * Matrix([1 for i in range(self.dim)])
        # Alternatively
        self._u = self.H_layer * Matrix([1 if i==0 else 0 for i in range(self.dim)])

        # Projector onto uniform ket subspace - normal to simplex
        self._pu = self._u * self._u.T
        self._qu = eye(self.dim) - self._pu

        # Operator G2
        self._g2 = self.gamma*self._pu + self.delta*self._qu

    def find(self, x0, m=None):
        """Find element x0 \in [0, ..., N-1] with m iterations of Grover's operator"""

        # Optimal number of iterations
        if m is None:
            m = self._m_hat #int(self.N**(.5)) # TODO fix constant - it's pi/4 or something

        self.xin = self._u
        self.xout = self._K(x0)**m * self.xin
        self.p = [abs(a)**2 for a in list(self.xout)]

        return self.xout

    def prob_plot(self, x0, m=None, ax=None):
        """Plot probs"""

        # Optimal number of iterations
        if m is None:
            m = self._m_hat #int(self.N**(.5)) # TODO fix constant - it's pi/4 or something

        if ax is None:
            fig, ax = plt.subplots()

        self.find(x0, m=m)

        ax.bar(x=range(self.dim), height=self.p, width=.9)

        ax.set_xlim(-.5, self.dim-.5)
        ax.set_ylim(0, 1.5)

        ax.set_xticks(list(range(self.dim)))

        ax.set_xlabel('k')
        ax.set_ylabel('P(k)')

        return ax

    def period_plot(self, x0=0, m_max=None, ax=None):
        """Plot probability of finding x0 vs. m."""

        if m_max is None:
            m_max = 4*self._m_hat

        if not hasattr(x0, '__len__'):
            x0 = [x0]

        if ax is None:
            fig, ax = plt.subplots()

        ax.axhline(y=0, ls='--', c='gray')
        ax.axhline(y=1, ls='--', c='gray')

        ax.set_xlabel('m')
        ax.set_ylabel(r'P[$x_{out}$ = $x_0$]')

        ax.set_title(r'$N$={}'.format(self.N)) #'    $x_0$={}'.format(self.N, x0))

        for _x0 in x0:
            K = self._K(_x0)
            xout = self._u
            ms = range(1,m_max+1)
            ps = []

            for m in ms:
                xout = K*xout
                ps += [abs(xout[_x0])**2]

            ax.scatter(ms, ps, s=30, label=str(_x0))#edgecolor=, facecolor='w', marker='o', s=30)

        ax.set_xticks(ms)
        ax.legend(title=r'$x_0$')

        return ax

    @property
    def _m_hat(self):
        return int((Rational(1,2) * (Rational(1,2) * pi / asin(1/sqrt(self.N)) - 1)).evalf())

    def _K(self, x0):
        """Grover kernel"""
        return self._g2*self._g1(x0)

    def _g1(self, x0):
        return self.alpha*self._px(x0) + self.beta*self._qx(x0)

    def _px(self, x0):
        return self._xket(x0) * self._xket(x0).T

    def _qx(self, x0):
        return eye(self.dim)-self._px(x0)

    def _xket(self, x0):
        return Matrix([1 if x0==i else 0 for i in range(self.dim)])

#
#class Oracle(cirq.Gate):
#    # copied from https://quantumcomputing.stackexchange.com/questions/4521/how-do-i-create-my-own-unitary-matrices-that-i-can-apply-to-a-circuit-in-cirq
#
#    """
#    Oracle gate returning 1 on secret state and 0 otherwise.
#    """
#
#    def __init__(self, secret_state):
#        """Sets secret state"""
#        self.secret_state = secret_state
#
#    def num_qubits(self):
#        return 3
#
#    def _apply_unitary_to_tensor_(self, target_tensor, available_buffer, axes):
#        """Define unitary implementing oracle"""
#        s = cirq.slice_for_qubits_equal_to(axes, self.secret_state)
#        target_tensor[s] *= -1
#        return target_tensor
#
#N = 5 # number of items in database
#x0 = 3 # distinguished element in database
#assert x0<=N, 'x_0 cannot be greater than N'
#
#n = int(np.ceil(np.log2(N))) # number of qbits needed
#m = int(N**(.5)) # optimal number of iterations
#
## Initialise qubits
#qb = cirq.LineQubit.range(N)
#
## Create circuit
#g = cirq.Circuit()
#
## Flip target register 0->1 - NB: NOT = X
#g.append(cirq.X(qb[-1]))
#
## Hadamard layer
#g.append(cirq.H.on(q) for q in qb)
#
## Oracle
#oracle = Oracle(x0)
#g.append(Oracle.on(q) for q in qb)
