#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file of the project
"""
import argparse
import numpy as np

from u_tilde_function import UTildeFunction
from visual_tools import plot_2d_function,plot_approx,plot_error
from mesh import Mesh
from f_function import FFunction
from solver import solve
from u_function import UFunction
from error_analysis import calc_l2_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--visualize", help="Plot the initial visualization", action='store_true')
    parser.add_argument('-m', "--mesh", help="Generate mesh", action='store_true')
    parser.add_argument('-s', "--solve", help="Starts the solver", action='store_true')
    args = parser.parse_args()

    if args.visualize:
        visualize_u_tilde()

    elif args.mesh:
        mesh = Mesh()
        mesh.generate_mesh(5,5)
        mesh.draw()

    elif args.solve:
        mesh = Mesh(15,15)
        f_function = FFunction()
        vertices, u = solve(mesh,f_function,accuracy=1.49e-1)
        plot_approx(vertices, u)
    else:
        h_tests = np.array([4,8,16])
        errors = np.zeros_like(h_tests,dtype=float)
        i = 0
        for d in np.nditer(h_tests):
            print("[Info] M="+str(d))
            mesh = Mesh(d, d)
            f_function = FFunction()
            vertices, u = solve(mesh, f_function, accuracy=1.49e1)
            u_func = UFunction(u, vertices)
            u_tilde_func = UTildeFunction()
            e = calc_l2_error(u_func,u_tilde_func)
            errors[i] = e
            print("[Info] L2 error for M="+str(d)+": "+str(errors[i]))
            i+=1
        plot_error(h_tests,errors)

def visualize_u_tilde():
    """
    Visualize u_tilde
    :return:
    """
    u = UTildeFunction()
    plot_2d_function(u,100000)

if __name__ == "__main__":
    main()