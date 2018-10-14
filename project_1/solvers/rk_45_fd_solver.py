#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Solves time dependant problems by integration using the RK45 solver
"""

import numpy as np

from scipy.integrate import RK45


def solve_dynamic_system(system, args, max_step, t_bound, x_0, t_0=0,bc_imposer = None, bc_args = None):
    """
    Solves a dynamical system using Dormand–Prince with 4th order error control and 5th order stepping "RK45".
    :param system: A callable dynamic system taking (t,x,J)
    :param args: Arguments to be passed to the system
    :param max_step: The maximum allowed length of a step
    :param t_bound: The time at which to stop integration
    :param x_0: The initial state
    :param t_0: The initial time
    :param bc_imposer: Callable that can be used to modify x every timestep in order to impose bc.
    :param bc_args: Args for the bc_imposer
    :return: An array of states and an array of timestamps
    """
    x = x_0
    t_arr = np.zeros((1, 1))
    x = np.expand_dims(x, axis=1)

    ivp = RK45(fun=lambda t, y: system(t, y, args), t0=t_0, y0=x_0, t_bound=t_bound, max_step=max_step, rtol=0.001,
               atol=1e-06,
               vectorized=False)

    while True:
        if (ivp.t >= t_bound):
            break
        ivp.step()
        if bc_imposer is not None:
            ivp.y = bc_imposer(ivp.y,ivp.t,bc_args)
        x = np.append(x, np.expand_dims(ivp.y, axis=1), axis=1)
        t_arr = np.append(t_arr, np.ones((1, 1)) * ivp.t, axis=1)

    print("[Info] Made " + str(t_arr.shape[1]) + " timesetps")

    return x, t_arr
