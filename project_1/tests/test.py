#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for the code"""

import unittest
import numpy as np

from project_1.functions.f_function import FFunction
from project_1.functions.u_tilde_function import UTildeFunction
from project_1.infrastructure.p1_reference_element import P1ReferenceElement
from project_1.infrastructure.affine_transformation import AffineTransformation


class TestCode(unittest.TestCase):
    """
    Useful test cases
    """

    def setUp(self):
        pass

    def test_f_function(self):
        """
        Tests the implemented f function
        :return:
        """
        f_test_instance = FFunction()
        self.assertAlmostEqual(f_test_instance.value((0.5, 0.3)), 0)
        self.assertAlmostEqual(f_test_instance.value((0.0, 0.0)), 0)
        self.assertAlmostEqual(f_test_instance.value((0.0, 1.0)), 0)
        self.assertAlmostEqual(f_test_instance.value((1.0, 0.0)), 0)
        self.assertAlmostEqual(f_test_instance.value((1.0, 1.0)), 0)

    def test_u_tile_function(self):
        """
        Tests the implemented u_tilde function
        :return:
        """

        def evaluate_lhs(x, u_tilde_test_instance):
            """
            Evaluates the left hand side of the PDE at a given position
            :param x: Coordinate to evaluate on
            :param u_tilde_test_instance: The solution to the PDE
            :return: The value of the LHS
            """
            v = u_tilde_test_instance.value(x) - u_tilde_test_instance.laplacian(x)
            return v

        u_tilde_test_instance = UTildeFunction()
        f_test_instance = FFunction()

        # Evaluate consistency with given solution
        x = (0, 0.5)
        self.assertAlmostEqual(u_tilde_test_instance.value(x), 1)

        # Evaluate correctness of solution regarding f
        n = 100
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(n ** 2)
        yv = yv.reshape(n ** 2)
        for i in range(n**2):
            self.assertAlmostEqual(evaluate_lhs((xv[i],yv[i]), u_tilde_test_instance) - f_test_instance.value((xv[i],yv[i])), 0)


        # Evaluate compliance with boundary conditions
        # Dirichlet
        bd = np.arange(0,1,0.01)
        for i in    np.nditer(bd):
            self.assertAlmostEqual(u_tilde_test_instance.value((i,0)), 0)
            self.assertAlmostEqual(u_tilde_test_instance.value((i, 1)), 0)

        # Neumann
        n = np.array([[-1, 0]]).T
        for i in    np.nditer(bd):
            self.assertAlmostEqual(np.asscalar(u_tilde_test_instance.gradient((0,i)).T.dot(n)), 0)
            self.assertAlmostEqual(np.asscalar(u_tilde_test_instance.gradient((1, i)).T.dot(n)), 0)

    def test_p1_reference_element(self):
        """
        Tests the implemented P1 reference element
        :return:
        """
        p1_reference_element = P1ReferenceElement()

        # Check nodal basis
        x = (0, 0)
        v = p1_reference_element.value(x)
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 0)
        self.assertEqual(v[2], 0)

        x = (1, 0)
        v = p1_reference_element.value(x)
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 1)
        self.assertEqual(v[2], 0)

        x = (0, 1)
        v = p1_reference_element.value(x)
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 0)
        self.assertEqual(v[2], 1)

        # Check sum properties
        n = 100
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(n ** 2)
        yv = yv.reshape(n ** 2)
        for i in range(n**2):
            if(xv[i]+yv[i]<=1):
                self.assertAlmostEqual(np.sum(p1_reference_element.value((xv[i],yv[i]))),1)


    def test_affine_transformation(self):
        """
        Tests the affine transformation
        :return:
        """

        affine_trafo = AffineTransformation()

        # Reference
        x0 = (0, 0)
        x1 = (1, 0)
        x2 = (0, 1)
        self.assertEqual(affine_trafo.get_determinant(x0, x1, x2), 1)

        # Invalid simplex
        x1 = x0
        self.assertEqual(affine_trafo.get_determinant(x0, x1, x2), 0)

        # Test case from assignment
        x0 = (1, 0)
        x1 = (3, 1)
        x2 = (3, 2)
        self.assertGreater(affine_trafo.get_determinant(x0, x1, x2), 0)

        #Test inverse
        j = affine_trafo.get_jacobian(x0, x1, x2)
        j_inv = affine_trafo.get_inverse_jacobian(x0, x1, x2)
        np.testing.assert_array_almost_equal(j_inv.dot(j),np.identity(2))

        v0 = np.array([[1, 0]]).T
        x = np.array([[1, 0]]).T
        x_new = j_inv.dot(x-v0)
        np.testing.assert_array_almost_equal(x_new,np.array([[0, 0]]).T)
        x = np.array([[3, 1]]).T
        x_new = j_inv.dot(x-v0)
        np.testing.assert_array_almost_equal(x_new,np.array([[1, 0]]).T)
        x = np.array([[3, 2]]).T
        x_new = j_inv.dot(x-v0)
        np.testing.assert_array_almost_equal(x_new,np.array([[0, 1]]).T)


if __name__ == '__main__':
    print("Starting unittest...")
    unittest.main()
