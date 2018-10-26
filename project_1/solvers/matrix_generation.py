"""
Generates the stiffness and mass matrices
"""

import numpy as np
import scipy.integrate as integrate
from project_1.utils.integration import gauss_legendre_reference

def generate_mass_matrix(accuracy, atraf, mesh, p1_ref, quadpack, triangles, varnr, vertices):
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
    return M


def generate_stiffness_matrix(accuracy, atraf, mesh, p1_ref, quadpack, triangles, varnr, vertices):
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
                else:
                    ans, err = gauss_legendre_reference(stiffness_matrix_integrant_fast,
                                                        args=(p1_ref, i, j, jinvt, result))
                K[tr_current.v[i], tr_current.v[j]] += np.abs(atraf.get_determinant()) * ans
    return K

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