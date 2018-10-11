#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Represents a triangle
"""

import numpy as np

class Triangle:
    """
    Represents a triangle
    """

    def __init__(self,v0,v1,v2,id):
        """
        Initializes a triangle
        :param v0: The id of vertex 0
        :param v1: The id of vertex 1
        :param v2: The id of vertex 2
        param id: Id of the triangle
        """
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.id = id

        self.v = np.zeros(3,dtype=int)
        self.v[0] = v0
        self.v[1] = v1
        self.v[2] = v2