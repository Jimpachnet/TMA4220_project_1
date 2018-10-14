#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tools to analyze the error of the solution
"""

import numpy as np
import scipy.integrate as integrate

from project_1.infrastructure.affine_transformation import AffineTransformation
from project_1.infrastructure.p1_reference_element import P1ReferenceElement
from project_1.utils.integration import gauss_legendre_reference
import project_1.infrastructure.mesh


def calc_l2_error(u_function, u_tilde_function):
    """
    Calculates the l2 norm of the error
    :param u_function: The approximated function
    :param u_tilde_function: The analytical solution
    :return: The L2 error
    """

    def integrant(y, x, u_function, u_tilde_function):
        return (u_function.value((x, y)) - u_tilde_function.value((x, y))) ** 2

    ans, err = integrate.dblquad(integrant, 0, 1, lambda x: 0, lambda x: 1, args=(u_function, u_tilde_function))

    return np.sqrt(ans)


def calc_l2_error_simplex_based(mesh, u_tilde_function, u):
    """
    Calculates the L2 error of the solution by adding it up over the individual simplices
    :param mesh: The mesh to compute on
    :param u_tilde_function: The analytical solution
    :param u: The solution array
    :return: The L2 error
    """

    def error_integrant_reference(y, x, p1_ref, u_function, j, v0_coord, det, u_vals):
        co = (x, y)
        xc = np.array([[x], [y]])
        x0 = np.array([[v0_coord[0]], [v0_coord[1]]])
        x_new = j.dot(xc) + x0
        trival = 0
        for i in range(3):
            trival += p1_ref.value(co)[i] * u_vals[i]
        return np.asscalar(trival - u_function.value((x_new[0], x_new[1]))) ** 2

    vertices = mesh.vertices
    triangles = mesh.triangles
    atraf = AffineTransformation()
    p1_ref = P1ReferenceElement()
    error = 0

    for n in range(len(mesh.triangles)):
        tr_current = mesh.triangles[n]
        v0_coord = (vertices[0, triangles[n].v0], vertices[1, triangles[n].v0])
        v1_coord = (vertices[0, triangles[n].v1], vertices[1, triangles[n].v1])
        v2_coord = (vertices[0, triangles[n].v2], vertices[1, triangles[n].v2])
        atraf.set_target_cell(v0_coord, v1_coord, v2_coord)

        x_min = np.min([v0_coord[0], v1_coord[0], v2_coord[0]])
        x_max = np.max([v0_coord[0], v1_coord[0], v2_coord[0]])
        y_min = np.min([v0_coord[1], v1_coord[1], v2_coord[1]])
        y_max = np.max([v0_coord[1], v1_coord[1], v2_coord[1]])
        jinvt = atraf.get_inverse_jacobian().T
        j = atraf.get_jacobian()
        u_vals = (u[triangles[n].v0], u[triangles[n].v1], u[triangles[n].v2])
        det = atraf.get_determinant()
        ans, err = gauss_legendre_reference(error_integrant_reference,
                                            args=(p1_ref, u_tilde_function, j, v0_coord, det, u_vals))
        error += ans
    return np.sqrt(error)
