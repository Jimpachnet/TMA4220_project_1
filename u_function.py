#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy import interpolate

class UFunction:
    """Class representing the approximated solution to the PDE

    Todo: Beautify
    """

    def __init__(self,u_values,supports):
        """
        Initialize the function
        :param u_values: values of u at supports
        :param supports: array of supports
        """
        self.u_values = u_values
        self.supports = supports

        self.f = interpolate.interp2d(supports[0,:], supports[1,:], u_values, kind='linear')

    def value(self,x):
        """
        Evaluates u(x) at x
        :param x: Tuple (x,y) of the coordinate at which the function should be evaluated
        :return: The value of u(x) at x
        """
        u = self.f(x[0],x[1])

        return u

