#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix


from project_1.infrastructure.p1_reference_element import P1ReferenceElement
from project_1.infrastructure.affine_transformation import AffineTransformation
from project_1.utils.integration import gauss_legendre_reference




def solve_forces(mesh, quadpack=False, accuracy=1.49e-05):
    vertices = mesh.vertices
    triangles = mesh.triangles
    varnr = mesh.supportsy * mesh.supportsx
    atraf = AffineTransformation()
    p1_ref = P1ReferenceElement()
    supports = 7
    print("[Info] Calculating stiffness matrix")
    K = generate_stiffness_matrix_new(accuracy, atraf, mesh, p1_ref, quadpack, triangles, varnr, vertices)

    plt.imshow(K, interpolation='nearest', cmap=plt.cm.ocean,
               extent=(0.5, np.shape(K)[0] + 0.5, 0.5, np.shape(K)[1] + 0.5))
    plt.colorbar()
    plt.show()


    print("[Info] Calculating linear form")
    b = generate_linear_form_new(accuracy, atraf, mesh, p1_ref, quadpack, supports, triangles, varnr, vertices)

    print(K)
    print(b)

    A = K
    # BC Dirichlet
    nr = np.shape(vertices)[1]
    for i in range(nr):
        if vertices[1, i] == -1 or vertices[1, i] == 1 or vertices[0, i] == -1 or vertices[0, i] == 1:
            A[i*2, :] = np.zeros((1, nr*2))
            A[i*2, i*2] = 1
            b[i*2] = 0
            A[i*2+1, :] = np.zeros((1, nr*2))
            A[i*2+1, i*2+1] = 1
            b[i*2+1] = 0

    # Solve system
    u = np.linalg.inv(A).dot(b)




    return vertices, u










def solve_forces_old(mesh, quadpack=False, accuracy=1.49e-05):
    vertices = mesh.vertices
    triangles = mesh.triangles
    varnr = mesh.supportsy * mesh.supportsx
    atraf = AffineTransformation()
    p1_ref = P1ReferenceElement()
    supports = 7

    # Stiffness Matrix
    print("[Info] Calculating stiffness matrix")
    K = generate_stiffness_matrix(accuracy, atraf, mesh, p1_ref, quadpack, triangles, varnr, vertices)

    # b
    print("[Info] Calculating linear form")
    b = generate_linear_form(accuracy, atraf, mesh, p1_ref, quadpack, supports, triangles, varnr, vertices)

    A = K

    # BC Dirichlet
    nr = np.shape(vertices)[1]
    for i in range(nr):
        if vertices[1, i] == -1 or vertices[1, i] == 1 or vertices[0, i] == -1 or vertices[0, i] == 1:
            A[i, :] = np.zeros((1, nr))
            A[i, i] = 1
            b[i] = 0

    # Solve system
    u = np.linalg.inv(A).dot(b)



    plt.imshow(K-K.T, interpolation='nearest', cmap=plt.cm.ocean,
               extent=(0.5, np.shape(A)[0] + 0.5, 0.5, np.shape(K-K.T)[1] + 0.5))
    plt.colorbar()
    plt.show()

    plt.imshow(K, interpolation='nearest', cmap=plt.cm.ocean,
               extent=(0.5, np.shape(A)[0] + 0.5, 0.5, np.shape(K)[1] + 0.5))
    plt.colorbar()
    plt.show()

    return vertices, u


def generate_linear_form(accuracy, atraf, mesh, p1_ref, quadpack, supports, triangles, varnr, vertices):
    b = np.zeros((varnr, 1))
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
        det = atraf.get_determinant()
        for i in range(3):
            if quadpack:
                ans, err = integrate.dblquad(b_integrant, x_min, x_max, lambda x: y_min, lambda x: y_max,
                                             epsabs=accuracy, epsrel=accuracy,
                                             args=(p1_ref, i, jinvt, v0_coord))
            else:
                ans, err = gauss_legendre_reference(b_integrant_reference,
                                                    args=(p1_ref, i, j, v0_coord, det), supports=supports)
            b[tr_current.v[i]] += ans
    return b

def generate_linear_form_new(accuracy, atraf, mesh, p1_ref, quadpack, supports, triangles, varnr, vertices):
    b = np.zeros((varnr*2, 1))
    for n in range(len(mesh.triangles)):
        tr_current = mesh.triangles[n]
        v0_coord = (vertices[0, triangles[n].v0], vertices[1, triangles[n].v0])
        v1_coord = (vertices[0, triangles[n].v1], vertices[1, triangles[n].v1])
        v2_coord = (vertices[0, triangles[n].v2], vertices[1, triangles[n].v2])
        atraf.set_target_cell(v0_coord, v1_coord, v2_coord)
        j = atraf.get_jacobian()
        det = atraf.get_determinant()
        for i in range(3):
            ans, err = gauss_legendre_reference(b_integrant_reference_1,
                                                    args=(p1_ref, i, j, v0_coord, det), supports=supports)
            b[tr_current.v[i]*2] += ans
            a1 = ans
            ans, err = gauss_legendre_reference(b_integrant_reference_2,
                                                args=(p1_ref, i, j, v0_coord, det), supports=supports)
            b[tr_current.v[i]*2+1] += ans

    return b




def b_integrant_reference_1(y, x, p1_ref, i, j, v0_coord, det):

    def f_function(x,y):
        Youngs_E_Modulus = 250 * 10 ^ 9
        v = 0.3
        fx = Youngs_E_Modulus/(1-v**2)*(-2*y**2-x**2+v*x**2-2*v*x*y-2*x*y+3-v)
        return fx

    co = (x, y)
    xc = np.array([[x], [y]])
    x0 = np.array([[v0_coord[0]], [v0_coord[1]]])
    x_new = j.dot(xc) + x0
    return np.asscalar(p1_ref.value(co)[i] * f_function(x_new[0], x_new[1])) * np.abs(det)

def b_integrant_reference_2(y, x, p1_ref, i, j, v0_coord, det):

    def f_function(x,y):
        Youngs_E_Modulus = 250 * 10 ^ 9
        v = 0.3
        fx = Youngs_E_Modulus/(1-v**2)*(-2*x**2-y**2+v*y**2-2*v*x*y-2*x*y+3-v)
        return fx

    co = (x, y)
    xc = np.array([[x], [y]])
    x0 = np.array([[v0_coord[0]], [v0_coord[1]]])
    x_new = j.dot(xc) + x0
    return np.asscalar(p1_ref.value(co)[i] * f_function(x_new[0], x_new[1])) * np.abs(det)











def b_integrant(y, x, p1_ref, i, jinvt, v0_coord):

    co = (x, y)
    xp = np.array([x - v0_coord[0], y - v0_coord[1]])
    x_tr = (jinvt.dot(xp)[0], jinvt.dot(xp)[1])
    val = p1_ref.value(x_tr)[i]
    return val * f_function.value(co)


def b_integrant_reference(y, x, p1_ref, i, j, v0_coord, det):

    def f_function(x,y):
        Youngs_E_Modulus = 250 * 10 ^ 9
        v = 0.3
        fx = Youngs_E_Modulus/(1-v**2)*(-2*y**2-x**2+v*x**2-2*v*x*y-2*x*y+3-v)
        return fx

    co = (x, y)
    xc = np.array([[x], [y]])
    x0 = np.array([[v0_coord[0]], [v0_coord[1]]])
    x_new = j.dot(xc) + x0
    return np.asscalar(p1_ref.value(co)[i] * f_function(x_new[0], x_new[1])) * np.abs(det)


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




                #START
                Youngs_E_Modulus = 250 * 10 ^ 9
                v = 0.3

                lame_lambda = Youngs_E_Modulus * v / ((1 + v) * (1 - 2 * v))
                K_ = Youngs_E_Modulus / (3 * (1 - 2 * v))
                mue = 3 * (K_ - Youngs_E_Modulus) / 2



                D_0 = np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])
                D = Youngs_E_Modulus/(1-v**2)*D_0


                B = np.zeros((3,1))
                B[0:2,:] = jinvt.T.dot(p1_ref.gradients(co)[:, i])
                B[2,0] = np.sum(jinvt.T.dot(p1_ref.gradients(co)[:, i]))

                Bt = np.zeros((3, 1))
                Bt[0:2,:] = jinvt.T.dot(p1_ref.gradients(co)[:, j])
                Bt[2,0] = np.sum(jinvt.T.dot(p1_ref.gradients(co)[:, j]))

                result = np.asscalar(B.T.dot(D).dot(Bt))


                #END



                if quadpack:
                    ans, err = integrate.dblquad(stiffness_matrix_integrant_fast, 0, 1, lambda x: 0, lambda x: 1,
                                                 epsabs=accuracy, epsrel=accuracy, args=(p1_ref, i, j, jinvt, result))
                else:
                    ans, err = gauss_legendre_reference(stiffness_matrix_integrant_fast,
                                                        args=(p1_ref, i, j, jinvt, result))
                K[tr_current.v[i], tr_current.v[j]] += np.abs(atraf.get_determinant()) * ans


    return K


def generate_stiffness_matrix_new(accuracy, atraf, mesh, p1_ref, quadpack, triangles, varnr, vertices):
    K = np.zeros((varnr*2, varnr*2))
    for n in range(len(mesh.triangles)):
        tr_current = mesh.triangles[n]
        v0_coord = (vertices[0, triangles[n].v0], vertices[1, triangles[n].v0])
        v1_coord = (vertices[0, triangles[n].v1], vertices[1, triangles[n].v1])
        v2_coord = (vertices[0, triangles[n].v2], vertices[1, triangles[n].v2])
        atraf.set_target_cell(v0_coord, v1_coord, v2_coord)
        jinvt = atraf.get_inverse_jacobian().T

        # START
        Youngs_E_Modulus = 250 * 10 ^ 9
        v = 0.3

        D_0 = np.array([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])
        D = Youngs_E_Modulus / (1 - v ** 2) * D_0

        #D_0_2 = np.array([[1, v/(1-v), 0], [v/(1-v), 1, 0], [0, 0, (1 - 2*v) / 2*(1-v)]])

        #D = Youngs_E_Modulus*(1-v)/((1+v)*(1-2*v))*D_0_2

        local_K = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                # In order to make calculation feasible
                co = (0.1, 0.1)

                B_i= np.zeros((3, 1))
                B_j = np.zeros((3, 1))

                if i % 2 == 0:
                    B_i[0,:] = jinvt.T.dot(p1_ref.gradients(co)[:, i//2])[0]
                    B_i[2, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i//2])[1]
                else:
                    B_i[1, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i//2])[1]
                    B_i[2, :] = jinvt.T.dot(p1_ref.gradients(co)[:, i//2])[0]
                if j % 2 == 0:
                    B_j[0,:] = jinvt.T.dot(p1_ref.gradients(co)[:, j//2])[0]
                    B_j[2, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j//2])[1]
                else:
                    B_j[1, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j//2])[1]
                    B_j[2, :] = jinvt.T.dot(p1_ref.gradients(co)[:, j//2])[0]

                #print("j:"+str(j))
                #print(j//2)

                result = np.asscalar(B_i.T.dot(D).dot(B_j))


                #END



                if quadpack:
                    ans, err = integrate.dblquad(stiffness_matrix_integrant_fast, 0, 1, lambda x: 0, lambda x: 1,
                                                 epsabs=accuracy, epsrel=accuracy, args=(p1_ref, i, j, jinvt, result))
                else:
                    #ans, err = gauss_legendre_reference(stiffness_matrix_integrant_fast,
                    #                                    args=(p1_ref, i, j, jinvt, result))
                    ans = result/2

                local_K[i,j] = np.abs(atraf.get_determinant()) * ans


        if ~np.isclose(np.linalg.det(local_K),0):
            print("ERROR")

        for i in range(3):
            for j in range(3):
                K[tr_current.v[i]*2, tr_current.v[j]*2] += local_K[i*2,j*2]
                K[tr_current.v[i] * 2 + 1, tr_current.v[j] * 2 + 1] += local_K[i * 2+1, j * 2+1]



    return K


def stiffness_matrix_integrant_fast(y, x, p1_ref, i, j, jinvt, result):
    if (x + y > 1):
        result = 0
    return result


def stiffness_matrix_integrant(y, x, p1_ref, i, j, jinvt):
    co = (x, y)
    return jinvt.dot(p1_ref.gradients(co)[:, i]).T.dot(jinvt.dot(p1_ref.gradients(co)[:, j]))