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


def gauss_legendre_reference(integrand, args,supports = 7):
    """
    Uses gauss legendre to integrate over a simplex reference cell in 2D
    :param integrand: The function to integrate
    :param args: Arguments to pass to the function
    :param supports: Number of supports to evaluate
    :return: The evaluation of the integral
    """
    # Todo Is different than in exercise!!!
    if supports == 7:
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
    elif supports == 4:
        value = 0
        argsp = barycentric_to_cartesian_reference(1 / 3, 1 / 3, 1 / 3) + args
        value += integrand(*argsp) * (-9/16)
        argsp = barycentric_to_cartesian_reference(1 / 5, 1 / 5, 1 / 3) + args
        value += integrand(*argsp) * (25/48)
        argsp = barycentric_to_cartesian_reference(1 / 5, 1 / 3, 1 / 5) + args
        value += integrand(*argsp) * (25/48)
        argsp = barycentric_to_cartesian_reference(1 / 3, 1 / 5, 1 / 5) + args
        value += integrand(*argsp) * (25/48)
        return value, 0
    elif supports == 3:
        value = 0
        argsp = barycentric_to_cartesian_reference(1 / 2, 1 / 2, 0) + args
        value += integrand(*argsp) * (1/3)
        argsp = barycentric_to_cartesian_reference(1 / 2,0, 1 / 2) + args
        value += integrand(*argsp) * (1/3)
        argsp = barycentric_to_cartesian_reference(0, 1 / 2, 1 / 2) + args
        value += integrand(*argsp) * (1/3)
        return value, 0
    elif supports == 1:
        argsp = barycentric_to_cartesian_reference(1 / 3, 1 / 3, 1/3) + args
        return integrand(*argsp), 0
        


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


def gauss_legendre_r1_test(evaluations,a,b):
    """
    Evaluates the integral \int_1^2 e^x dx using the Gauss-Legendre quadrature
    :param evaluations: Number of evaluations from 1 to 4
    :param a: Lower bound
    :param b: Upper bound
    :return: The evaluation of the integral
    """
    xi = (a+b)/2
    l = np.abs(a-b)


    if evaluations == 1:
        return (np.e**xi)*l
    elif evaluations == 2:
        v = (np.e**(xi+l*(np.sqrt(3)/6)))*0.5*l
        v += (np.e ** (xi - l * (np.sqrt(3) / 6))) * 0.5 * l
        return v
    elif evaluations == 3:
        weights = np.array([(5/18)*l,(5/18)*l,(8/18)*l])
        points = np.array([xi+l*(np.sqrt(15)/10),xi-l*(np.sqrt(15)/10),xi])
        return np.sum( weights*np.e**points)
    elif evaluations == 4:
        weights = np.array([(18-np.sqrt(30))/36*l,(18-np.sqrt(30))/36*l,(18+np.sqrt(30))/36*l,(18+np.sqrt(30))/36*l])/2
        points = np.array([xi+l*((np.sqrt(525+70*np.sqrt(30)))/70),xi-l*((np.sqrt(525+70*np.sqrt(30)))/70),xi+l*((np.sqrt(525-70*np.sqrt(30)))/70),xi-l*((np.sqrt(525-70*np.sqrt(30)))/70)])
        return np.sum(weights * np.e ** points)