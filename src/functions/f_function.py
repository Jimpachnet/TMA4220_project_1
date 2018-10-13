#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class FFunction:
    """Class representing the inhomogenous part of the PDE

    Todo: Beautify
    """

    def value(self,x):
        """
        Evaluates f(x) at x
        :param x: Tuple (x,y) of the coordinate at which the function should be evaluated
        :return: The value of f(x) at x
        """
        f = (2*np.pi**2+1)*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])

        return f