#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements the P1 reference element
"""

import numpy as np

class P1ReferenceElement:
    """
    Class representing the P1 reference element
    """

    def value(self,x):
        """
        Returns the value of the three shape functions at x
        :param x: A tuple of the (x,y) coordinate
        :return: An array with three values of the shape functions
        """

        p_0 = 1-x[0]-x[1]
        p_1 = x[0]
        p_2 = x[1]

        return np.array([p_0,p_1,p_2])

    def gradients(self,x):
        """
        Calculates the gradient of the reference element
        :param x: The position at which the gradient should be calculated
        :return: Matrix 2x3 containing the gradients
        """
        grad  = np.matrix('-1 -1 0; -1 0 -1')

        return grad
