#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class UTildeFunctionDynamic:
    """Class representing the analytical solution to the PDE

    Todo: Beautify
    """

    def value(self,x,t):
        """
        Evaluates u_tilde(x) at x
        :param x: Tuple (x,y) of the coordinate at which the function should be evaluated
        :param t: The time to evaluate u at.
        :return: The value of u_tilde(x,t) at x,t
        """
        theta = np.pi/4
        u_tilde = np.e**(-t)*np.sin(x[0]*np.cos(theta)+x[1]*np.cos(theta))

        return u_tilde
