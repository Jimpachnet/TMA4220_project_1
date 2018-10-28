#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements solver for 2D wave problem
"""
import numpy as np
import scipy.integrate as integrate

from project_1.solvers.matrix_generation import generate_mass_matrix, generate_stiffness_matrix
from project_1.infrastructure.p1_reference_element import P1ReferenceElement
from project_1.infrastructure.affine_transformation import AffineTransformation
from project_1.utils.integration import gauss_legendre_reference
from project_1.solvers.rk_45_fd_solver import solve_dynamic_system
from scipy.interpolate import LinearNDInterpolator, interp1d


def solve_wave_dynamic(mesh, t_end, t_0=0, timestep=0.01, quadpack=False, accuracy=1.49e-05):
    """
    Solves the Helmholtz problem under fixed BC.
    :param mesh: The mesh to operate on
    :param f_function: The inhomogenous right hand side
    :param quadpack: Should the Fortran quadpack package be used to integrate numerically
    :param accuracy: The accuracy for quadpack
    :return: An ND interpolator
    """

    c = 0.5

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
    K*=c**2

    b = np.zeros((varnr, 1))

    A = -np.linalg.inv(M).dot(K)

    # "Window" BC Dirichlet
    nr = np.shape(vertices)[1]


    bm = np.linalg.inv(M).dot(b)

    u = np.zeros((varnr, 1))

    print("[Info] Solving system in time domain")

    u0 = np.ones_like(u[:, 0]) * 0

    v0 = np.ones_like(u[:, 0]) * 0

    x0 = np.zeros((2 * varnr))

    x0[0:varnr] = u0
    x0[varnr:] = v0

    J = np.zeros((2 * varnr, 2 * varnr))
    J[0:varnr, varnr:] = np.eye(varnr)
    J[varnr:, 0:varnr] = A

    def system(t, y, args):
        J = args[0]
        b = args[1]
        dy = J.dot(y)
        varn = b.shape[0]
        dy[varn:] +=b[:,0]
        return J.dot(y)

    def bc_imposer(y,t,args):
        varnr = args[0]
        vertices = args[1]

        for i in range(varnr):
            if vertices[1, i] == 1:
                if t<(1/3):
                    y[i] = np.sin(3*np.pi*t)*0.2
                else:
                    y[i] = 0

        for i in range(varnr):
            if vertices[1, i] <= 0.5 and vertices[0,i]<0.5:
                y[i] = 0

        for i in range(varnr):
            if vertices[0, i] == 0:
                y[i] = 0

        for i in range(varnr):
            if vertices[0, i] == 1:
                y[i] = 0

        return y

    x, t_arr = solve_dynamic_system(system, (J,bm), timestep, t_end, x0,bc_imposer=bc_imposer,bc_args=(varnr,vertices))

    print("[Info] Generating interpolator")
    u = x[0:varnr]
    t_arr = np.squeeze(t_arr)
    f = interp1d(t_arr, u)

    return f


def mass_matrix_integrant(y, x, p1_ref, i, j):
    co = (x, y)
    return p1_ref.value(co)[i] * p1_ref.value(co)[j]


def stiffness_matrix_integrant_fast(y, x, p1_ref, i, j, jinvt, result):
    if (x + y > 1):
        result = 0
    return result


def stiffness_matrix_integrant(y, x, p1_ref, i, j, jinvt):
    co = (x, y)
    return jinvt.dot(p1_ref.gradients(co)[:, i]).T.dot(jinvt.dot(p1_ref.gradients(co)[:, j]))
