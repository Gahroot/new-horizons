"""Baseline: distance from every point to its nearest neighbour.

Straightforward but memory-hungry: it materialises the full n x n distance
matrix before taking row minima.
"""

import math


def nearest_neighbor_distances(points):
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi, zi = points[i]
        for j in range(n):
            xj, yj, zj = points[j]
            matrix[i][j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
    result = []
    for i in range(n):
        result.append(min(matrix[i][j] for j in range(n) if j != i))
    return result
