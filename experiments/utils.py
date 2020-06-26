"""
Distances and more.
"""

from functools import reduce
from sympy import Rational, eye, sqrt, simplify, sin, cos, pi
from sympy.physics.matrices import msigma

def qubit(theta, phi=0, r=1):
    """Qubit in the parametrization:

        rho = 1/2 * (I + a*sigma)

    with a = r*(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    """

    a = r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)
    rho = Rational(1,2) * (eye(2) + reduce(lambda x, y: x+y, [a[i]*msigma(1+i) for i in range(3)]))

    return simplify(rho) 

def trace_distance(rho, sigma):
    """Trace distance between density matrices `rho` and `sigma`"""
    return Rational(1,2) * reduce(lambda x, y: x+y, map(lambda kv: kv[1]*abs(kv[0]), (rho-sigma).eigenvals().items()))

def fidelity(rho, sigma):
    """Fidelity between density matrices `rho` and `sigma`"""
    # TODO is this computing the principal square root?
    A = (rho)**(1/2)*(sigma)**(1/2)
    return simplify(((A.conjugate().T * A)**(1/2)).trace()**2)

def bures(rho, sigma):
    """Bures metric"""
    return sqrt(2*(1-sqrt(fidelity(rho,sigma))))