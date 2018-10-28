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
from project_1.infrastructure.mesh import Mesh
from matplotlib.tri import Triangulation, LinearTriInterpolator


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

    def error_integrant_reference(y, x, p1_ref, u_function, j, v0_coord, det, fz):
        co = (x, y)
        xc = np.array([[x], [y]])
        x0 = np.array([[v0_coord[0]], [v0_coord[1]]])
        x_new = j.dot(xc) + x0
        trival =  fz(x_new[0],x_new[1])
        return np.asscalar(trival - u_function.value((x_new[0], x_new[1]))) ** 2



    atraf = AffineTransformation()
    p1_ref = P1ReferenceElement()
    error = 0

    m_ref = Mesh(88,88)
    triangles = m_ref.triangles
    trianglesarr = np.zeros((len(mesh.triangles), 3))

    i = 0
    for triangle in mesh.triangles:
        trianglesarr[i, :] = triangle.v
        i += 1




    triObj = Triangulation(mesh.vertices[0,:], mesh.vertices[1,:],trianglesarr)
    fz = LinearTriInterpolator(triObj, u[:,0])
    vertices = m_ref.vertices
    for n in range(len(m_ref.triangles)):
        v0_coord = (vertices[0, triangles[n].v0], vertices[1, triangles[n].v0])
        v1_coord = (vertices[0, triangles[n].v1], vertices[1, triangles[n].v1])
        v2_coord = (vertices[0, triangles[n].v2], vertices[1, triangles[n].v2])
        atraf.set_target_cell(v0_coord, v1_coord, v2_coord)

        j = atraf.get_jacobian()
        det = atraf.get_determinant()
        ans, err = gauss_legendre_reference(error_integrant_reference,
                                            args=(p1_ref, u_tilde_function, j, v0_coord, det, fz))
        error += ans*np.abs(det)
    return np.sqrt(error)
