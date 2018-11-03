#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates a simplex mesh based on the input
"""

import numpy as np

from project_1.infrastructure.triangle import Triangle
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


class Mesh:
    """
    Represents a mesh
    """

    def __init__(self, sx, sy, h=2, w=2):
        """
        Initialize the mesh
        :param sx: Number of support points in x
        :param sy: Number of support points in y
        :param h: Height of the rectangle
        :param w: Width of the rectangle
        """
        self.generate_mesh(sx, sy, h, w)

    def generate_mesh(self, supportsx, supportsy, height=2, width=2, x0 = -1, y0 = -1):
        """
        Generates a simplex mesh in 2D on a rectangle
        :param supportsx: Number of support points in x
        :param supportsy: Number of support points in y
        :param height: Height of the rectangle
        :param width: Width of the rectangle
        :return: The mesh
        """

        h_x = width / (supportsx - 1)
        h_y = height / (supportsy - 1)

        trianglelist = []
        id = 0
        tri_id = 0
        vertices = np.zeros((2, supportsx * supportsy))

        for y in range(supportsy):
            for x in range(supportsx):
                vertices[:, id] = np.array([x * h_x+x0, y * h_y+y0])
                if y < supportsy - 1:
                    if x > 0:
                        t_upper = Triangle(id + supportsx, id + supportsx - 1, id, tri_id)
                        trianglelist.append(t_upper)
                        tri_id += 1
                    if x < supportsx - 1:
                        t_lower = Triangle(id, id + 1, id + supportsx, tri_id)
                        trianglelist.append(t_lower)
                        tri_id += 1
                id += 1

        self.triangles = trianglelist
        self.vertices = vertices
        self.supportsx = supportsx
        self.supportsy = supportsy

    def draw(self):
        """
        Draws the mesh
        :return:
        """
        vertices = self.vertices
        triangles = self.triangles

        colorticker = False
        fig, ax = plt.subplots()

        for i in range(len(triangles)):
            v0 = vertices[:, triangles[i].v0]
            v1 = vertices[:, triangles[i].v1]
            v2 = vertices[:, triangles[i].v2]
            pos = np.array([v0, v1, v2])
            plt.text((v0[0] + v1[0] + v2[0]) / 3, (v0[1] + v1[1] + v2[1]) / 3, "K" + str(i))
            if colorticker:
                polygon = Polygon(pos, True, linewidth=5, color='blue')
            else:
                polygon = Polygon(pos, True, linewidth=5, color='red')
            colorticker = not colorticker
            ax.add_patch(polygon)

        for i in range(np.shape(vertices)[1]):
            plt.text(vertices[0, i], vertices[1, i], "V" + str(i))

        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
