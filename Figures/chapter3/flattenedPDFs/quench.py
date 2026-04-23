# -*- coding: utf-8 -*-
"""
quench.py

Created on Tue Sep  2 16:23:05 2025
v1.0

@author: sane
Contains various protocols for time varying diffusivity, coordinate transformations, etc.
"""
import numpy as np
import scipy

def invert(f, y, x_0=0, dx=1e-3, max_iters = 100):
    """
    Given y = f(x) where f is a monotonic function, find x = f_inv(y) using Newton's method.

    Parameters
    ----------
    f : function
        monotonic function of x.
    y : numeric
        Position to evaluate inverse function at.

    Returns
    -------
    f^{-1}(y), where f^{-1}(f(x))=x.
    """
    F = lambda x: y-f(x)
    F_prime = lambda x: (F(x+dx)-F(x))/dx
    
    x = x_0
    for i in range(max_iters):
        x -= F(x)/F_prime(x)
    return x
        
def integrate(f, t, t_0=0):
    """Integrate a scalar function with vector bounds."""
    return np.vectorize(lambda x: scipy.integrate.quad(f, t_0, x)[0])(t)

class QuenchProtocol(object):
    """Class for defining quench functions. A quench function is always scaled by the bath temperature -- so the temperature at time t is given by k_BT(t) = k_BT_b*h(t)."""
    
    def __init__(self, a):
        self.a=a
        return None
    def set_a(self, new_a):
        """Reset the value of the temperature ratio a = (k_BT_h/k_BT_b - 1)."""
        self.a=new_a
    def t_transformed(self, t):
        """Find the transformed time t_1 as a function of the proper time in the new coordinates."""
        return integrate(self.h, t)
    def t(self, t_1, x0=0):
        """Invert the transformed time function (THIS ASSUMES THE QUENCH PROTOCOL h IS POSITIVE EVERYWHERE) to get the time as a function of transformed time."""
        return np.vectorize(lambda T: scipy.optimize.fsolve(lambda y: self.t_transformed(y)-T, x0=x0))(t_1)
    def h_1(self, t_1):
        """Evaluate the composition h(t(t_1)) -- effectively, the quench function in the new coordinates, which we need to divide the force by."""
        return self.h(self.t(t_1))
        
class InstantaneousQuench(QuenchProtocol):
    """Child class of the quench protocol for a trivial quench that instantaneously drops to 1."""
    
    def __init__(self, a=None):
        """Initialise the class. a doesn't really matter here since h(t) = 1 for all t > 0, but we do need it so that we can inherit from QuenchProtocol."""
        super().__init__(a)
    def h(self, t):
        """Trivial quench function."""
        return 1

class ExponentialQuench(QuenchProtocol):
    """Child class of the quench protocol for Newtonian cooling -- exponential decay to bath temperature."""
    
    def __init__(self, a, tau):
        """In addition to the temperature ratio parameter a we also need a time constant tau that controls the decay rate."""
        self.tau = tau
        super().__init__(a)
    def h(self, t):
        """Exponential quench protocol -- equal to a+1 = k_BT_h/k_BT_b at t=0 and 1 when t->inf."""
        return self.a*np.exp(-t/self.tau)+1
