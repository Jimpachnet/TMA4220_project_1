#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements solver for 2D Helmholtz problem
"""
import numpy as np
import scipy.integrate as integrate

from mesh import Mesh
from f_function import FFunction
from p1_reference_element import P1ReferenceElement
from affine_transformation import AffineTransformation
from visual_tools import plot_approx


def solve(mesh,f_function,accuracy = 1.49e-05):
    """
    Solves the Helmholtz problem under fixed BC.
    :param mesh: The mesh to operate on
    :param f_function: The inhomogenous right hand side
    :return:
    """

    vertices = mesh.vertices
    triangles = mesh.triangles
    varnr = mesh.supportsy*mesh.supportsx
    atraf = AffineTransformation()
    p1_ref = P1ReferenceElement()

    #Mass matrix
    print("[Info] Calculating mass matrix")
    M = np.zeros((varnr,varnr))

    for n in range(len(mesh.triangles)):
        tr_current = mesh.triangles[n]
        v0_coord = (vertices[0,triangles[n].v0],vertices[1,triangles[n].v0])
        v1_coord = (vertices[0, triangles[n].v1], vertices[1, triangles[n].v1])
        v2_coord = (vertices[0, triangles[n].v2], vertices[1, triangles[n].v2])
        atraf.set_target_cell(v0_coord,v1_coord,v2_coord)
        for i in range(3):
            for j in range(3):
                ans, err = integrate.dblquad(mass_matrix_integrant, 0, 1, lambda x: 0, lambda x: 1, epsabs=accuracy, epsrel=accuracy, args=(p1_ref, i, j))
                M[tr_current.v[i],tr_current.v[j]] += atraf.get_determinant()*ans

    #Stiffness Matrix
    print("[Info] Calculating stiffness matrix")
    K = np.zeros((varnr,varnr))
    for n in range(len(mesh.triangles)):
        tr_current = mesh.triangles[n]
        v0_coord = (vertices[0,triangles[n].v0],vertices[1,triangles[n].v0])
        v1_coord = (vertices[0, triangles[n].v1], vertices[1, triangles[n].v1])
        v2_coord = (vertices[0, triangles[n].v2], vertices[1, triangles[n].v2])
        atraf.set_target_cell(v0_coord,v1_coord,v2_coord)
        jinvt = atraf.get_inverse_jacobian().T
        for i in range(3):
            for j in range(3):
                #In order to make calculation feasible
                co = (0.1, 0.1)
                result = jinvt.dot(p1_ref.gradients(co)[:, i]).T.dot(jinvt.dot(p1_ref.gradients(co)[:, j]))

                ans, err = integrate.dblquad(stiffness_matrix_integrant_fast, 0, 1, lambda x: 0, lambda x: 1, epsabs=accuracy, epsrel=accuracy, args=(p1_ref, i, j,jinvt,result))
                #ans2, err2 = integrate.dblquad(stiffness_matrix_integrant, 0, 1, lambda x: 0, lambda x: 1, epsabs=accuracy, epsrel=accuracy, args=(p1_ref, i, j,jinvt))
                K[tr_current.v[i],tr_current.v[j]] += atraf.get_determinant()*ans

    #b
    print("[Info] Calculating linear form")
    b = np.zeros((varnr,1))
    for n in range(len(mesh.triangles)):
        tr_current = mesh.triangles[n]
        v0_coord = (vertices[0,triangles[n].v0],vertices[1,triangles[n].v0])
        v1_coord = (vertices[0, triangles[n].v1], vertices[1, triangles[n].v1])
        v2_coord = (vertices[0, triangles[n].v2], vertices[1, triangles[n].v2])
        atraf.set_target_cell(v0_coord,v1_coord,v2_coord)

        x_min = np.min([v0_coord[0],v1_coord[0],v2_coord[0]])
        x_max = np.max([v0_coord[0], v1_coord[0], v2_coord[0]])
        y_min = np.min([v0_coord[1], v1_coord[1], v2_coord[1]])
        y_max = np.max([v0_coord[1], v1_coord[1], v2_coord[1]])
        jinvt = atraf.get_inverse_jacobian().T
        for i in range(3):
            ans, err = integrate.dblquad(b_integrant, x_min, x_max, lambda x: y_min, lambda x: y_max, epsabs=accuracy/10, epsrel=accuracy/10, args=(p1_ref, i,f_function,jinvt,v0_coord))
            b[tr_current.v[i]] += ans

    #Solve system
    A = K+M
    u = np.linalg.inv(A).dot(b)
    plot_approx(vertices,u)


def mass_matrix_integrant(y,x,p1_ref,i,j):
    co = (x,y)
    return p1_ref.value(co)[i]*p1_ref.value(co)[j]

def stiffness_matrix_integrant_fast(y,x,p1_ref,i,j,jinvt,result):
    if(x+y>1):
        result = 0
    return result


def stiffness_matrix_integrant(y, x, p1_ref, i, j, jinvt):
    co = (x, y)
    return jinvt.dot(p1_ref.gradients(co)[:,i]).T.dot(jinvt.dot(p1_ref.gradients(co)[:,j]))

def b_integrant(y,x,p1_ref,i,f_function,jinvt,v0_coord):
    co = (x,y)
    xp = np.array([x-v0_coord[0],y-v0_coord[1]])
    x_tr = (jinvt.dot(xp)[0],jinvt.dot(xp)[1])
    val = p1_ref.value(x_tr)[i]
    return val*f_function.value(co)

