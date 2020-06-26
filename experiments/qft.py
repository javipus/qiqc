"""
QFT and HSP on arbitrary finite abelian groups. I'm following this paper:

   [Anurag Sahay, The Hidden Subgroup Problem](https://anuragsahay.github.io/cs682.pdf)
"""

# TODO implement HSP algorithm. It has the following steps:
#
# 0. Initialise source and target to 0
# 1. Apply DFT to source to produce uniform superposition:
#
#   \frac{1}{\sqrt{|G|}}\sum_{g \in G} |g>|0>
#
# -- Notice how DFT(|0>) being a uniform superposition is exactly what you would expect from uncertainty considerations, super-localised deltas being dual to scattered plane waves, etc.
#
# 2. Apply f oracle to produce
#
#   \frac{1}{\sqrt{|G|}\sum_{g\in G}}|g>|f(g)>
#
# 3. Apply DFT to source again to produce (this algebra is not as straightforward, make sure you get it):
#
#   \frac{1}{\sqrt{|H^{\perp}|}}\sum_{t\in T}\phi_t |H^{\perp}>|f(t)>
#
# where
#
#   H^{\perp} = \{g\in G: \forall h\in H\quad \chi_g(h) = 1\} <-- notice how this reduces to orthogonality in the exponent
#
#   T = G/H <-- representatives of H cosets - we can do this bc G is abelian so H is normal
#
#   \phi_t = \sum_{g\in G} \chi_g(h) |g><g| <-- phase shift operator
#
# 4. Measure source register to get random element of H^{\perp}
# 5. Repeat k times - the probability of obtaining a generating set for H^{\perp} is 1-2**(-k)
# 6. Compute H as H^{\perp\perp}. There is an efficient algorithm for that in http://arxiv.org/pdf/quant-ph/0411037v1.pdf

from functools import reduce
from itertools import product
from numbers import Number

import numpy as np

from sympy import I, pi, exp, sqrt, eye, flatten, Rational, Matrix
from sympy.physics.quantum import TensorProduct as TP

class FiniteAbelianGroup:
    """
    Every finite abelian group is isomorphic to the product of cyclic groups

        Z_N1 X ... Z_Nk

    for integers (N1, ..., Nk)
    """

    def __init__(self, *Ns):
        """
        Create group $\oplus_k Z_{N_k}$.

        @param Ns: List of integers. Order of each factor in the product.
        """

        self._Ns = Ns
        self.dim = len(Ns)
        self.order = reduce(lambda x,y: x*y, self._Ns)
        self._basis = [self._element([1 if i==j else 0 for i in range(self.dim)]) for j in range(self.dim)]
        self._omega = [exp(2*I*pi*Rational(1,n)) for n in self._Ns]

    def HSP(self, f):
        """Solve the hidden subgroup problem by calling oracle f."""

        # STEP 0
        # Initialise register
        # NB: I don't know the size of the second register beforehand
        # An upper bound is |G|, which happens e.g. when H = {e} and every coset is different
        # This means I need log2|G| qubits - let's go with that
        n_src = int(np.ceil(np.log2(self.order)))
        n_tgt = int(np.ceil(np.log2(self.order)))

        # Ket |0>^{n} in the computational basis has coordinates [1] + [0]*(2**(n-1)) in tensor basis
        ket = Matrix([1 if i==0 else 0 for i in range(2**(n_src+n_tgt))])

        # STEP 1
        # QFT on source qubits, identity on target
        DFT = TP(self._DFT, eye(2**n_tgt))
        ket = DFT * ket

        # STEP 2
        # TODO I don't know how to construct the oracle
        # what I've implemented is NOT UNITARY
        # Oracle unitary U_f
        Uf = sqrt(Rational(1,self.order)) * Matrix([[1 if f(g)==h else 0 for h in self.elements] for g in self.elements])
        ket = TP(eye(2**n_src), Uf) * ket

        # STEP 3
        # Fourier transform again
        ket = DFT * ket

        # STEP 4
        # Measurement
        # Probability vector after marginalising over target register
        # Cumbersome conversion to np.array to apply sum method
        # TODO isn't there a native sympy function to do this?
        p = (abs(np.array(ket.reshape(2**n_tgt, 2**n_src)).astype(np.float64))**2).sum(axis=0)
#        idx = np.random.choice(range(self.order), p=p)
#        elements = self.elements
#        state = next(elements)
#
#        for _ in range(idx):
#            state = next(elements)
#
        #return ket #.reshape(2**n_tgt, 2**n_src)
        return Uf

    def QFT(self, ket):
        """Apply QFT to ket in basis given by self.elements."""

        if len(ket) != self.order:
            raise ValueError('Length of ket ({}) and group order ({}) must coincide!'.format(len(ket), self.order))
        return self._DFT * Matrix(ket)

    def chi(self, g):
        """
        Character mapping an arbitrary element h to:

            $chi_g(h) = \Pi\limits_{j=1}^{k} \omega_{j}^{g_j h_j}

        where $\oemga_j$, $g_j$ and $h_j$ are the components of these elements in the standard basis and.
        """

        return lambda h: reduce(lambda x,y: x*y, [oj**(gj*hj) for oj, gj, hj in zip(self._omega, g, h)])

    def _measure(self, ket):
        """Sample group according to probabilities encoded in ket."""
        return np.random.choice(self.elements, p=[abs(c) for c in flatten(ket)])

    def _toKet(self, g):
        """Map element to ket in the tensor basis."""
        return Matrix([1 if g==h else 0 for h in self.elements])

    def _fromKet(self, ket):
        """Map ket in the tensor basis to element."""
        # TODO handle non-basis ket - should raise exception, right?
        return [g for g, c in zip(self.elements, flatten(ket)) if c][0]

    def _T(self, t):
        """Translation operator."""
        return Matrix([[1 if h+t==g else 0 for h in self.elements] for g in self.elements])

    def _P(self, j):
        """Phase-shift operator."""
        return Matrix([[self.chi(j)(g) if g==h else 0 for h in self.elements] for g in self.elements])

    @property
    def _DFT(self):
        # TODO what happens when |G| != 2^n?
        """Discrete Fourier Transform |G|-by-|G| matrix in the ket basis."""
        return sqrt(Rational(1,self.order))*Matrix([[self.chi(g)(h) for g in self.elements] for h in self.elements])

    @property
    def elements(self):
        """Iterator of elements."""
        for item in product(*[range(n) for n in self._Ns]):
            yield self._element(item)

    @classmethod
    def _element_factory(cls, Ns):
        class Element:

            def __init__(self, x, Ns):
                if not len(x) == len(Ns): raise ValueError('Length of element and group factors must coincide!')

                self._x = [_x%n for _x, n in zip(x, Ns)]
                self._Ns = Ns

            def __len__(self):
                return len(self._x)

            def __getitem__(self, k):
                return self._x[k]

            def __iter__(self):
                for item in self._x:
                    yield item

            def __eq__(self, other):
                """Element equality."""

                if hasattr(other, '_Ns'):
                    if self._Ns != other._Ns:
                        raise TypeError('Can only add elements of the same group!')

                return all([(x-y)%n==0 for x, y, n in zip(self, other, self._Ns)])

            def __ne__(self, other): return not self.__eq__(other)

            def __mod__(self, n):
                """Component-wise modulo integer n."""
                return Element([x%n for x in self._x], self._Ns)

            def __add__(self, other):
                """Component-wise addition modulo N1, ..., Nk"""

                if hasattr(other, '_Ns'):
                    if self._Ns != other._Ns:
                        raise TypeError('Can only add elements of the same group!')

                return Element([(x+y)%n for x, y, n in zip(self, other, self._Ns)], self._Ns)

            def __radd__(self, other): return self.__add__(other)

            def __mul__(self, other):
                """Multiplication by scalar"""

                if not isinstance(other, Number):
                    raise TypeError('Can only multiply by scalars!')

                return Element([(other*x)%n for x, n in zip(self._x, self._Ns)], self._Ns)

            def __rmul__(self, other): return self.__mul__(self, other)

            def __repr__(self):
                return repr(self._x)

            def __str__(self):
                return str(self._x)

        return lambda x: Element(x, Ns)

    def _element(self, x):
        return FiniteAbelianGroup._element_factory(self._Ns)(x)

    def __call__(self, *xs):
        """Return element with xs components."""
        return self._element(xs)

    def __repr__(self):
        return ' X '.join(['Z_{}'.format(n) for n in self._Ns])

    def __str__(self):
        return ' X '.join(['Z_{}'.format(n) for n in self._Ns])
