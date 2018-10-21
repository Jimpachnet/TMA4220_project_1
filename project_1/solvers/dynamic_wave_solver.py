#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements solver for 2D wave problem
"""
import numpy as np
import scipy.integrate as integrate

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
    M = np.zeros((varnr, varnr))

    for n in range(len(mesh.triangles)):
        tr_current = mesh.triangles[n]
        v0_coord = (vertices[0, triangles[n].v0], vertices[1, triangles[n].v0])
        v1_coord = (vertices[0, triangles[n].v1], vertices[1, triangles[n].v1])
        v2_coord = (vertices[0, triangles[n].v2], vertices[1, triangles[n].v2])
        atraf.set_target_cell(v0_coord, v1_coord, v2_coord)
        for i in range(3):
            for j in range(3):
                if quadpack:
                    ans, err = integrate.dblquad(mass_matrix_integrant, 0, 1, lambda x: 0, lambda x: 1, epsabs=accuracy,
                                                 epsrel=accuracy, args=(p1_ref, i, j))
                else:
                    ans, err = gauss_legendre_reference(mass_matrix_integrant, args=(p1_ref, i, j))
                M[tr_current.v[i], tr_current.v[j]] += np.abs(atraf.get_determinant()) * ans

    # Stiffness Matrix
    print("[Info] Calculating stiffness matrix")
    K = np.zeros((varnr, varnr))
    for n in range(len(mesh.triangles)):
        tr_current = mesh.triangles[n]
        v0_coord = (vertices[0, triangles[n].v0], vertices[1, triangles[n].v0])
        v1_coord = (vertices[0, triangles[n].v1], vertices[1, triangles[n].v1])
        v2_coord = (vertices[0, triangles[n].v2], vertices[1, triangles[n].v2])
        atraf.set_target_cell(v0_coord, v1_coord, v2_coord)
        jinvt = atraf.get_inverse_jacobian().T
        for i in range(3):
            for j in range(3):
                # In order to make calculation feasible
                co = (0.1, 0.1)
                result = jinvt.T.dot(p1_ref.gradients(co)[:, i]).T.dot(jinvt.T.dot(p1_ref.gradients(co)[:, j]))
                if quadpack:
                    ans, err = integrate.dblquad(stiffness_matrix_integrant_fast, 0, 1, lambda x: 0, lambda x: 1,
                                                 epsabs=accuracy, epsrel=accuracy, args=(p1_ref, i, j, jinvt, result))
                    # ans2, err2 = integrate.dblquad(stiffness_matrix_integrant, 0, 1, lambda x: 0, lambda x: 1, epsabs=accuracy, epsrel=accuracy, args=(p1_ref, i, j,jinvt))
                else:
                    ans, err = gauss_legendre_reference(stiffness_matrix_integrant_fast,
                                                        args=(p1_ref, i, j, jinvt, result))
                K[tr_current.v[i], tr_current.v[j]] += np.abs(atraf.get_determinant()) * ans
    K*=c**2

    # Leakage at (0,0)
    # K[0,:] *=0
    # K[0,0] = 1

    b = np.zeros((varnr, 1))

    A = -np.linalg.inv(M).dot(K)

    # "Window" BC Dirichlet
    nr = np.shape(vertices)[1]


    bm = np.linalg.inv(M).dot(b)

    u = np.zeros((varnr, 1))

    print("[Info] Solving system in time domain")

    u0 = np.ones_like(u[:, 0]) * 0
    #for i in range(varnr):
    #    if vertices[0, i] == 1 or vertices[0, i] == 0 or vertices[1, i] == 1 or vertices[1, i] == 0:
    #        u0[i] = 0
    #    elif (vertices[0, i] - 0.5) ** 2 <= 0.0002 ** 2 and (vertices[1, i] - 0.5) ** 2 <= 0.0002 ** 2:
    #        u0[i] = 1 - ((vertices[0, i] - 0.5) ** 2 + (vertices[1, i] - 0.5) ** 2) ** (0.5)

    v0 = np.ones_like(u[:, 0]) * 0

    # Non homogenous initial condition
    # u0 = u[:,0]
    # for i in range(varnr):
    #    u0[i] = reference_function.value(vertices[:,i],0)

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
        #for i in range(varnr):
        #    if vertices[1, i] == 0:
        #        y[i] = 0

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

    # Todo: Beautify
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


def b_integrant(y, x, p1_ref, i, f_function, jinvt, v0_coord):
    co = (x, y)
    xp = np.array([x - v0_coord[0], y - v0_coord[1]])
    x_tr = (jinvt.dot(xp)[0], jinvt.dot(xp)[1])
    val = p1_ref.value(x_tr)[i]
    return val * f_function.value(co)


def b_integrant_reference(y, x, p1_ref, i, f_function, j, v0_coord, det):
    co = (x, y)
    xc = np.array([[x], [y]])
    x0 = np.array([[v0_coord[0]], [v0_coord[1]]])
    x_new = j.dot(xc) + x0
    return np.asscalar(p1_ref.value(co)[i] * f_function.value((x_new[0], x_new[1]))) * np.abs(det)
