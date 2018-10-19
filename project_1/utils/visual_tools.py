#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tools to help visualizing functions in 2D"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import tqdm
from matplotlib import cm
from project_1.functions.u_function import UFunction
from project_1.functions.u_tilde_function import UTildeFunction
from project_1.functions.u_function_tilde_dynamic import UTildeFunctionDynamic
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection



def plot_2d_function(function_object, supports=100):
    """
    Plots a 2d function
    :param function_object: The function to be plottet. has to provide a .value(x) function.
    :param supports: Discrete supports at which the function should be evaluated. If an integer is given, the function
    will generate an equidistant grid with the given number of supports. If a list or array of points is given, those will be used for evaluation.
    """

    if type(supports) is int:
        perside = int(np.sqrt(supports))
        h = 1 / (perside - 1)
        coordinate_list = []
        for i in range(perside):
            for j in range(perside):
                coordinate_list.append((i * h, j * h))
        # Makes problems because of how the plot function works

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
            Z[ni, nj] = function_object.value((i, j))
            nj += 1
        ni += 1

    # Plot the surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Y, X, Z, cmap=cm.plasma,
                           linewidth=0, antialiased=True)
    # Todo X and Y axis seem to be turned
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


def plot_dynamic_2d_function_from_int_plain(lnd, t_end, t0=0, timestep=0.01, supports=100):
    """
     Use Linear ND interpolator https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator
     to plot dynamic function
    :param lnd: Linear ND interpolator
    :param t_end: Time at which the simulation should end
    :param t0: Start time
    :param timestep: Time delta between evaluations
    :param supports: Nuber of supports
    """

    if type(supports) is int:
        perside = int(np.sqrt(supports))
        h = 1 / (perside - 1)
        coordinate_list = []
        for i in range(perside):
            for j in range(perside):
                coordinate_list.append((i * h, j * h))
        # Makes problems because of how the plot function works

        x = np.linspace(0, 1, num=perside)
        y = np.linspace(0, 1, num=perside)

    else:
        raise NotImplementedError("To be implemented")

    t_arr = np.arange(t0, t_end, timestep)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='u', artist='Test',
                    comment='Works')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure()

    with writer.saving(fig, "wave_plain.mp4", dpi=300):
        for t in range(np.shape(t_arr)[0]):
            print("[Info] Plotting timestep " + str(t) + "/" + str(np.shape(t_arr)[0]))

            vals = np.zeros((perside, perside))
            for i in range(perside):
                for j in range(perside):
                    vals[i, j] = lnd((t_arr[t], i, j))

            plt.imshow(vals)

            writer.grab_frame()
            plt.gcf().clear()


def plot_dynamic_2d_function_from_int(lnd, t_end, mesh, t0=0, timestep=0.01, minv=0, maxv=1, filename="pde",
                                      supports=100):
    """
     Use Linear ND interpolator https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator
     to plot dynamic function
    :param lnd: Linear ND interpolator
    :param t_end: Time at which the simulation should end
    :param mesh: Mesh object
    :param t0: Start time
    :param timestep: Time delta between evaluations
    :param minv: Minimal value of z axis
    :param maxv: Maximal value of z axis
    :param filename: Name of the generated video
    :param supports: Nuber of supports
    """

    verticies = mesh.vertices

    if type(supports) is int:
        perside = int(np.sqrt(supports))
        h = 1 / (perside - 1)
        coordinate_list = []
        for i in range(perside):
            for j in range(perside):
                coordinate_list.append((i * h, j * h))
        # Makes problems because of how the plot function works

        x = np.linspace(0, 1, num=perside)
        y = np.linspace(0, 1, num=perside)

    else:
        raise NotImplementedError("To be implemented")

    t_arr = np.arange(t0, t_end, timestep)

    x = verticies[0, :]
    y = verticies[1, :]
    triangles = np.zeros((len(mesh.triangles), 3))
    i = 0
    for triangle in mesh.triangles:
        triangles[i, :] = triangle.v
        i += 1

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='u', artist='Test',
                    comment='Works')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure()
    print("[Info] Generating animation frames...")
    with writer.saving(fig, filename + ".mp4", dpi=300):
        for t in tqdm.tqdm(range(np.shape(t_arr)[0])):
            ax = fig.gca(projection='3d')
            cs = ax.plot_trisurf(x, y, np.squeeze(lnd(t_arr[t])), cmap=cm.plasma)
            cbar = plt.colorbar(cs)
            plt.xlabel("x")
            plt.ylabel("y")
            cbar.set_clim(minv, maxv)
            ax.set_zlim(minv, maxv)
            plt.title(r'$u(x),\ t=$' + str(round(t_arr[t], 3)) + "s")
            ax.view_init(30, -70)
            writer.grab_frame()
            plt.gcf().clear()


def plot_dynamic_2d_function(dynamic_function_object, t_end, t0=0, timestep=0.01, supports=100):
    """
    Plots a time dependant function
    :param dynamic_function_object: Time dependant function
    :param t_end: Time at which the simulation should end
    :param t0: Start time
    :param timestep: Time delta between evaluations
    :param supports: nuber of supports
    """
    if type(supports) is int:
        perside = int(np.sqrt(supports))
        h = 1 / (perside - 1)
        coordinate_list = []
        for i in range(perside):
            for j in range(perside):
                coordinate_list.append((i * h, j * h))
        # Makes problems because of how the plot function works

        x = np.linspace(0, 1, num=perside)
        y = np.linspace(0, 1, num=perside)

    else:
        raise NotImplementedError("To be implemented")

    t_arr = np.arange(t0, t_end, timestep)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='u', artist='Test',
                    comment='Works')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure()

    with writer.saving(fig, "heat_hom3d.mp4", dpi=300):
        for t in range(np.shape(t_arr)[0]):
            print("[Info] Plotting timestep " + str(t) + "/" + str(np.shape(t_arr)[0]))
            ni = 0
            for i in x:
                nj = 0
                for j in y:
                    prr = (i, j)
                    Z[ni, nj] = dynamic_function_object.value(prr, t_arr[t])
                    nj += 1
                ni += 1

            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(Y, X, Z, cmap=cm.plasma,
                                   linewidth=0, antialiased=True)
            # Todo X and Y axis seem to be turned
            cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
            cbar.set_clim(0, 1)
            plt.xlabel("x")
            plt.ylabel("y")

            ax.set_zlim(0, 1)
            plt.title(r'$u(x),\ t=$' + str(round(t_arr[t], 3)) + "s")
            ax.view_init(30, -70)

            writer.grab_frame()
            plt.gcf().clear()

    with writer.saving(fig, "heat_hom.mp4", dpi=300):
        for t in range(np.shape(t_arr)[0]):
            print("[Info] Plotting timestep " + str(t) + "/" + str(np.shape(t_arr)[0]))
            ni = 0
            for i in x:
                nj = 0
                for j in y:
                    prr = (i, j)
                    Z[ni, nj] = dynamic_function_object.value(prr, t_arr[t])
                    nj += 1
                ni += 1

            cs = plt.contourf(Y, X, Z, cmap=cm.plasma)
            cbar = plt.colorbar(cs)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(r'$u(x)$')
            writer.grab_frame()
            plt.gcf().clear()


def plot_approx(vertices, u):
    """
    Plots an approximate solution
    :param vertices: The vertices array
    :param u: The solution
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(vertices[0, :], vertices[1, :], u)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r'$u(x)$')
    ax.view_init(30, -70)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(vertices[0, :], vertices[1, :], u)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r'$u(x)$')
    ax.view_init(0, -90)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.scatter(vertices[0, :], vertices[1, :], u)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r'$u(x)$')
    ax.view_init(90, 0)
    plt.show()

    calu = UFunction(u, vertices)

    plot_2d_function(calu, 1000)


def plot_triangulated_helmholtz(mesh, u):
    """
    Plots the triangulated solution
    :param mesh: The mesh
    :param u: The solution
    """
    vertices = mesh.vertices
    x = vertices[0, :]
    y = vertices[1, :]

    triangles = np.zeros((len(mesh.triangles), 3))

    i = 0
    for triangle in mesh.triangles:
        triangles[i, :] = triangle.v
        i += 1
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    cs = ax.plot_trisurf(x, y, np.squeeze(u), cmap=cm.plasma)
    cbar = plt.colorbar(cs)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(r'$u(x)$')
    plt.show()


def plot_error(trials, errors, errors_app):
    """
    Plot error over different mesh sizes
    :param trials: Used mesh sizes
    :param errors: calculated error
    :param errors_app: Errors calculated by an approximation method
    """

    plt.plot(trials, errors)
    plt.plot(trials, errors_app)
    plt.xlabel("M")
    plt.ylabel("L2 error")
    plt.legend(['Quadpack', 'Gauss-Legendre'])
    plt.grid()
    plt.show()


def show_matrix(matrix):
    """
    Visualizes a matrix
    :param matrix: The matrix to visualize
    """
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()


def visualize_u_tilde():
    """
    Visualize u_tilde
    """
    u = UTildeFunction()
    plot_2d_function(u, 100000)


def visualize_u_tilde_dynamic():
    """
    Visualize u_tilde_dynamic
    """
    u = UTildeFunctionDynamic()
    plot_dynamic_2d_function(u, t_end=5, t0=0, timestep=0.1, supports=1000)

def visualize_nodal_basis_old():
    """
    Makes plots for the report
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    x = [1,0,0]
    y = [0,1,0]
    z = [0,0,1]
    verts = [list(zip(x, y, z))]
    print(verts)
    ax.add_collection3d(Poly3DCollection(verts), zs='z')
    plt.show()




def vis_all():
    """
    Create plots for report
    :return:
    """

    #visualize_nodal_basis()
    visualize_Gauss_Legendre_1d()

def visualize_nodal_basis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [1, 0, 0, 0]
    y = [0, 1, 0, 0]
    z = [0, 0, 1, 0]

    vertices = [[0, 0, 0], [1, 0, 3], [0, 2, 3], [1, 2, 3]]

    tupleList = list(zip(x, y, z))

    poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
    ax.scatter(x, y, z)
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='blue', linewidths=1, alpha=0.5))
    ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=2, linestyles=':'))
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

def visualize_Gauss_Legendre_1d():
    a = 1
    b = 2

    vals = np.zeros(4)
    xi = (a+b)/2
    l = np.abs(a-b)

    #1
    vals[0] = (np.e**xi)*l

    #2
    vals[1] = (np.e**(xi+l*(np.sqrt(3)/6)))*0.5*l
    vals[1] += (np.e ** (xi - l * (np.sqrt(3) / 6))) * 0.5 * l

    #3
    weights = np.array([(5/18)*l,(5/18)*l,(8/18)*l])
    points = np.array([xi+l*(np.sqrt(15)/10),xi-l*(np.sqrt(15)/10),xi])
    vals[2] = np.sum( weights*np.e**points)


    #4
    weights = np.array([(18-np.sqrt(30))/36*l,(18-np.sqrt(30))/36*l,(18+np.sqrt(30))/36*l,(18+np.sqrt(30))/36*l])
    points = np.array([xi+l*((np.sqrt(525+70*np.sqrt(30)))/70),xi-l*((np.sqrt(525+70*np.sqrt(30)))/70),xi+l*((np.sqrt(525-70*np.sqrt(30)))/70),xi-l*((np.sqrt(525-70*np.sqrt(30)))/70)])
    vals[3] = np.sum(weights * np.e ** points)/2

    print(weights)

    corr = np.e**2-np.e**1

    error = (vals-corr)**2

    plt.plot(np.log(error))
    plt.show()