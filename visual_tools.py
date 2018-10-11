#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tools to help visualizing functions in 2D"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from u_function import UFunction

def plot_2d_function(function_object,supports=100):
    """
    Plots a 2d function
    :param function_object: The function to be plottet. has to provide a .value(x) function.
    :param supports: Discrete supports at which the function should be evaluated. If an integer is given, the function
    will generate an equidistant grid with the given number of supports. If a list or array of points is given, those will be used for evaluation.
    :return:
    """

    if type(supports) is int:
        perside = int(np.sqrt(supports))
        h = 1/(perside-1)
        coordinate_list = []
        for i in range(perside):
            for j in range(perside):
                coordinate_list.append((i*h,j*h))
        #Makes problems because of how the plot function works

        x = np.linspace(0, 1, num=perside)
        y = np.linspace(0, 1, num=perside)

    else:
        raise NotImplementedError("To be implemented")


    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    ni = 0
    for i in x:
        nj = 0
        for j in y:
            Z[ni,nj] = function_object.value((i,j))
            nj+=1
        ni+=1


    # Plot the surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Y, X, Z, cmap=cm.plasma,
                           linewidth=0, antialiased=True)
    #Todo X and Y axis seem to be turned
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r'$u(x)$')
    ax.view_init(30, -70)
    plt.show()

    cs = plt.contourf(Y, X, Z, cmap=cm.plasma)
    cbar = plt.colorbar(cs)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r'$u(x)$')
    plt.show()

def plot_approx(vertices,u):
    """
    Plots an approximate solution
    :param vertices: The vertices array
    :param u: The solution
    :return:
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(vertices[0,:], vertices[1,:], u)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r'$u(x)$')
    ax.view_init(30, -70)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(vertices[0,:], vertices[1,:], u)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r'$u(x)$')
    ax.view_init(0, -90)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(vertices[0,:], vertices[1,:], u)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r'$u(x)$')
    ax.view_init(90,0)
    plt.show()

    calu = UFunction(u,vertices)

    plot_2d_function(calu,1000)




