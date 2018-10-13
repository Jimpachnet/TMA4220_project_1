#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import interpolate

class UFunctionDynamic:
    """Class representing the solution to the PDE

    Todo: Super quick and dirty code!!! Fix!
    """

    def __init__(self,time,support,u):
        """
        Initializes the function
        :param time: The time vector
        :param support: The support points
        :param u: The value vector
        """

        self.time = time
        self.support = support
        self.u = u



    def value(self,x,t):
        """
        Evaluates u(x) at x
        :param x: Tuple (x,y) of the coordinate at which the function should be evaluated
        :param t: The time to evaluate u at.
        :return: The value of u_tilde(x,t) at x,t
        """

        time = self.time
        supports = self.support
        u = self.u

        if t in time:
            timeindex = np.where(time == t)
            f = interpolate.interp2d(supports[0, :], supports[1, :], u[:,timeindex], kind='linear')
            u = f(x[0], x[1])
            return u

