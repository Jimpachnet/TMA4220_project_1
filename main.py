#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file of the project
"""
import argparse

from u_tilde_function import UTildeFunction
from visual_tools import plot_2d_function
from mesh import Mesh
from f_function import FFunction
from solver import solve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--visualize", help="Plot the initial visualization", action='store_true')
    parser.add_argument('-m', "--mesh", help="Generate mesh", action='store_true')
    parser.add_argument('-s', "--solve", help="Starts the solver", action='store_true')
    args = parser.parse_args()

    if args.visualize:
        visualize_u_tilde()

    if args.mesh:
        mesh = Mesh()
        mesh.generate_mesh(5,5)
        mesh.draw()

    if args.solve:
        mesh = Mesh(10,10)
        f_function = FFunction()
        solve(mesh,f_function,accuracy=1.49e-03)

def visualize_u_tilde():
    """
    Visualize u_tilde
    :return:
    """
    u = UTildeFunction()
    plot_2d_function(u,100000)

if __name__ == "__main__":
    main()