#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file of the project.
"""
import argparse
import numpy as np

from project_1.functions.u_tilde_function import UTildeFunction
from project_1.functions.u_function_tilde_dynamic import UTildeFunctionDynamic
from project_1.utils.visual_tools import *
from project_1.infrastructure.mesh import Mesh
from project_1.functions.f_function import FFunction
from project_1.solvers.solver_helmholtz import solve_helmholtz
from project_1.functions.u_function import UFunction
from project_1.utils.error_analysis import calc_l2_error
from project_1.solvers.dynamic_solver import solve_dynamic
from project_1.solvers.dynamic_wave_solver import solve_wave_dynamic
from project_1.functions.u_function_dynamic import UFunctionDynamic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--visualize", help="Plot the initial visualization", action='store_true')
    parser.add_argument('-vd', "--visualizedynamic", help="Plot the initial visualization", action='store_true')
    parser.add_argument('-m', "--mesh", help="Generate mesh", action='store_true')
    parser.add_argument('-s', "--solve", help="Starts the solver", action='store_true')
    parser.add_argument('-sd', "--solvedynamic", help="Starts the dynamic solver", action='store_true')
    parser.add_argument('-w', "--wave", help="Starts the dynamic solver for the wave equation", action='store_true')
    args = parser.parse_args()

    if args.visualize:
        visualize_u_tilde()

    elif args.mesh:
        mesh = Mesh(5,5)
        mesh.draw()

    elif args.solve:
        mesh = Mesh(25,25)
        f_function = FFunction()
        vertices, u = solve_helmholtz(mesh, f_function, accuracy=1.49e-1)
        plot_triangulated_helmholtz(vertices,mesh, u)
    elif args.solvedynamic:
        mesh = Mesh(10, 10)
        u_ref = UTildeFunctionDynamic()
        lnd = solve_dynamic(mesh,u_ref,0.2,t_0=0,timestep=0.01)
        plot_dynamic_2d_function_from_int(lnd,0.2, t0 = 0,timestep = 0.01,supports = 100)
    elif args.wave:
        mesh = Mesh(5,5)
        lnd = solve_wave_dynamic(mesh, 0.5, t_0=0, timestep=0.1)
        plot_dynamic_2d_function_from_int(lnd, 0.5, t0=0, timestep=0.1,minv=-0.5,maxv=0.5 ,supports=1000)
    elif args.visualizedynamic:
        visualize_u_tilde_dynamic()
    else:
        h_tests = np.array([4,8,16])
        errors = np.zeros_like(h_tests,dtype=float)
        i = 0
        for d in np.nditer(h_tests):
            print("[Info] M="+str(d))
            mesh = Mesh(d, d)
            f_function = FFunction()
            vertices, u = solve_helmholtz(mesh, f_function, accuracy=1.49e-1)
            u_func = UFunction(u, vertices)
            u_tilde_func = UTildeFunction()
            e = calc_l2_error(u_func,u_tilde_func)
            errors[i] = e
            print("[Info] L2 error for M="+str(d)+": "+str(errors[i]))
            i+=1
        plot_error(h_tests,errors)


if __name__ == "__main__":
    main()