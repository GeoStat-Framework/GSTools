#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:34:59 2018

@author: muellese
"""
import numpy as np


def unstructured(field, bin_edges, x, y=None, z=None):
    if bin_edges.shape[0] < 2:
        raise ValueError('len(bin_edges) too small')
    if x.shape[0] != field.shape[0]:
        raise ValueError('len(x) = {0} != len(f) = {1} '.
                         format(x.shape[0], field.shape[0]))
    # create a vector from the optional coordinates
    vector = []
    if y is not None:
        if y.shape[0] != field.shape[0]:
            raise ValueError('len(y) = {0} != len(f) = {1} '.
                             format(y.shape[0], field.shape[0]))
        vector.append(y)
    if z is not None:
        if z.shape[0] != field.shape[0]:
            raise ValueError('len(z) = {0} != len(f) = {1} '.
                             format(z.shape[0], field.shape[0]))
        vector.append(z)
    # calculate all field value differences and square it
    field_diff_quad = np.subtract.outer(field, field)**2
    # calculate all distances
    distance = np.subtract.outer(x, x)**2
    for coord in vector:
        distance += np.subtract.outer(coord, coord)**2
    distance **= 0.5
    # hack. don't include comparison of point with it self (diag is 0)
    # if bin_edges[0] = 0
    distance[np.diag_indices_from(distance)] = -1
    # calculate bin number
    bin_no = len(bin_edges) - 1
    variogram = np.zeros(bin_no, dtype=float)
    # iterate over all bins
    for i in range(bin_no):
        bin_mask = np.logical_and(distance >= bin_edges[i],
                                  distance < bin_edges[i+1])
        counts = np.sum(bin_mask)
        if counts > 0:
            variogram[i] = np.sum(field_diff_quad[bin_mask])/counts/2.0
    return variogram
