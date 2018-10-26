#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements solver for dynamic 2D problem
"""
import numpy as np

from project_1.infrastructure.p1_reference_element import P1ReferenceElement
from project_1.infrastructure.affine_transformation import AffineTransformation
from project_1.solvers.rk_45_fd_solver import solve_dynamic_system
from scipy.interpolate import interp1d
from project_1.solvers.matrix_generation import generate_mass_matrix, generate_stiffness_matrix


def solve_dynamic(mesh, reference_function, t_end, t_0=0, timestep=0.01, quadpack=False, accuracy=1.49e-05):
    """
    Solves the dynamic problem under fixed BC.
    :param mesh: The mesh to operate on
    :param reference_function: The function for the initial condition
    :param f_function: The inhomogenous right hand side
    :param quadpack: Should the Fortran quadpack package be used to integrate numerically
    :param accuracy: The accuracy for quadpack
    :return: A ND interpolator
    """

    vertices = mesh.vertices
    triangles = mesh.triangles
    varnr = mesh.supportsy * mesh.supportsx
    atraf = AffineTransformation()
    p1_ref = P1ReferenceElement()

    # Mass matrix
    print("[Info] Calculating mass matrix")
    M = generate_mass_matrix(accuracy, atraf, mesh, p1_ref, quadpack, triangles, varnr, vertices)

    # Stiffness Matrix
    print("[Info] Calculating stiffness matrix")
    K = generate_stiffness_matrix(accuracy, atraf, mesh, p1_ref, quadpack, triangles, varnr, vertices)

    b = np.zeros((varnr))
    nr = np.shape(vertices)[1]
    A = -np.linalg.inv(M).dot(K)


    t_arr = np.arange(t_0, t_end, timestep)
    nrtsteps = np.shape(t_arr)[0]

    u = np.zeros((varnr, nrtsteps))

    print("[Info] Solving system in time domain")
    u0 = np.ones_like(u[:, 0]) * 0.7

    def system(t, y, args):
        J = args[0]
        b = args[1]
        return J.dot(y) + b

    def bc_imposer(y,t,args):
        varnr = args[0]
        vertices = args[1]
        for i in range(varnr):
            if vertices[1, i] == 0:
                y[i] = 0

        for i in range(varnr):
            if vertices[1, i] == 1:
                #if vertices[0, i] > 0.3 and vertices[0, i] < 0.7:
                y[i] = 1
        return y

    x, t_arr = solve_dynamic_system(system, (A,np.linalg.inv(M).dot(b)), timestep, t_end, u0,bc_imposer=bc_imposer,bc_args=(varnr,vertices))

    # Todo: Beautify
    print("[Info] Generating interpolator")
    t_arr = np.squeeze(t_arr)

    f = interp1d(t_arr, x)

    return f







