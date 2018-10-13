#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tools to analyze the error of the solution
"""

import numpy as np
import scipy.integrate as integrate

def calc_l2_error(u_function,u_tilde_function):
    """
    Calculates the l2 norm of the error
    :param u_function: The approximated function
    :param u_tilde_function: The analytical solution
    :return: The L2 error
    """
    def integrant(y,x,u_function,u_tilde_function):

        return (u_function.value((x,y))-u_tilde_function.value((x,y)))**2

    ans, err = integrate.dblquad(integrant, 0, 1, lambda x: 0, lambda x: 1,  args=(u_function,u_tilde_function))

    return np.sqrt(ans)
