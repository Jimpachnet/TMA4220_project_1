#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file of the project
"""
import argparse

from u_tilde_function import UTildeFunction
from visual_tools import plot_2d_function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--visualize", help="Plot the initial visualization", action='store_true')
    args = parser.parse_args()
    if args.visualize:
        visualize_u_tilde()


def visualize_u_tilde():
    """
    Visualize u_tilde
    :return:
    """
    u = UTildeFunction()
    plot_2d_function(u,100000)

if __name__ == "__main__":
    main()