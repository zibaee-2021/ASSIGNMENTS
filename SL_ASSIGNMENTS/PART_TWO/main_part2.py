import python_functions_part2 as pf2
import numpy as np


if __name__ == '__main__':
    # Question 6: generate random dataset and visualise like Figure 1 in question sheet.
    X, y = pf2.generate_random_h()
    # generate 100 equidistant coordinates in 2 dimensions. Use np.meshgrid to for cartesian coordinates.
    a, b = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    A, B = np.meshgrid(a, b)
    i_axis_positions = 100
    j_axis_positions = 100
    y_grid = np.zeros((i_axis_positions, j_axis_positions))

    for i in range(i_axis_positions):
        for j in range(j_axis_positions):
            dists = pf2.calculate_distances_between(grid_point0=A[i][j], grid_point1=B[i][j], S=X)
            indices_of_closest_3 = pf2.get_closest_neighbours(dists)
            majority_class = pf2.get_majority_class_of_neighbours(indices_of_closest_3, class_labels=y)
            y_grid[i][j] = majority_class

    pf2.visualise_meshgrid(A, B, y_grid, X, y)
    pass





