#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements integration functions
"""

import numpy as np


def gauss_legendre_S(a, b, c, d, integrand, args):
    """
    Uses gauss legendre to integrate a function on R2
    :param a: The lower bound of x
    :param b: The upper bound of x
    :param c: The lower bound of y
    :param d: The upper bound of y
    :param integrand: The function to integrate
    :param args: Arguments to pass to the function
    :return: The evaluation of the integral
    """
    raise NotImplementedError()


def gauss_legendre_reference(integrand, args):
    """
    Uses gauss legendre to integrate over a simplex reference cell in 2D
    :param integrand: The function to integrate
    :param args: Arguments to pass to the function
    :return: The evaluation of the integral
    """
    # Todo Is different than in exercise!!!
    value = 0
    argsp = barycentric_to_cartesian_reference(1 / 3, 1 / 3, 1 / 3) + args
    value += integrand(*argsp) * 9 / 80

    min = False
    for i in range(6):
        if (min):
            a = (6 - np.sqrt(15)) / 21
        else:
            a = (6 + np.sqrt(15)) / 21
        argsp = barycentric_to_cartesian_reference(a, a, 1 - 2 * a) + args
        if (min):
            value += integrand(*argsp) * ((155 - np.sqrt(15)) / 2400)
        else:
            value += integrand(*argsp) * ((155 + np.sqrt(15)) / 2400)
        min = not min
    return value, 0


def barycentric_to_cartesian_reference(l1, l2, l3):
    """
    Converts barycentric coordinates to cartesian coordinates on the reference simplex
    :param l1: First barycentric coordinate
    :param l2: Second barycentric coordinate
    :param l3: Third barycentric coordinate
    :return: (y,x)
    """

    x = (l2) / (l1 + l2 + l3)
    y = (l3) / (l1 + l2 + l3)

    return (y, x)
