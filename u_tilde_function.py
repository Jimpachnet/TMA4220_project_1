#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class UTildeFunction:
    """Class representing the analytical solution to the PDE

    Todo: Beautify
    """

    def value(self,x):
        """
        Evaluates u_tilde(x) at x
        :param x: Tuple (x,y) of the coordinate at which the function should be evaluated
        :return: The value of u_tilde(x) at x
        """
        u_tilde = np.cos(np.pi*x[0])*np.sin(np.pi*x[1])

        return u_tilde

    def gradient(self,x):
        """
        Evaluates the gradient of u_tilde(x) at x
        :param x: Tuple (x,y) of the coordinate at which the gradient should be evaluated
        :return: Gradient at x represented as collumn vector
        """

        return np.array([-np.pi*np.sin(np.pi*x[1])*np.sin(np.pi*x[0]), np.pi*np.cos(np.pi*x[0])*np.cos(np.pi*x[1])]).T

    def laplacian(self,x):
        """
        Evaluates the laplacian of u_tilde(x) at x
        :param x: Tuple (x,y) of the coordinate at which the laplacian should be evaluated
        :return: Laplacian of u_tilde at x
        """

        return -np.pi**2*np.sin(np.pi*x[1])*np.cos(np.pi*x[0])-np.pi**2*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])