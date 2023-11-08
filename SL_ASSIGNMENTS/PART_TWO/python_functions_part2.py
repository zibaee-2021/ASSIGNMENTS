import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def generate_random_h(num_of_samples: int = 100) -> tuple:
    X = np.random.uniform(low=0, high=1, size=(100, 2))
    num_of_trials, prob_of_success = 1, 0.5
    y = np.random.binomial(n=num_of_trials, p=prob_of_success, size=num_of_samples)
    return X, y


def calculate_distances_between(grid_point0, grid_point1, S):
    """
    Calculate Euclidean distance between a point on the grid (akin to coordinates) and every data point in the dataset.
    :param grid_point0: A single data point, from one array of np.meshgrid() points.
    :param grid_point1: A single data point, from one array of np.meshgrid() points.
    :param S: Dataset. 2D numpy array, e.g. shape (100,2) for 100 centres.
    :return: Distances from given grid coordinates to each centre in given dataset.
    """
    distances = np.zeros(100)
    for i, X_centre in enumerate(S):
        distances[i] = np.sqrt((grid_point0 - X_centre[0]) ** 2 + (grid_point1 - X_centre[1]) ** 2)
    return distances


def get_closest_neighbours(dists, v=3):
    indices_of_lowest3 = np.argsort(dists)[:v]
    return indices_of_lowest3


def get_majority_class_of_neighbours(indices_of_closest_v, class_labels):
    closest_v_classes = class_labels[indices_of_closest_v]
    counts_of_0_1 = np.bincount(closest_v_classes)
    majority_class = np.argmax(counts_of_0_1)
    return majority_class


def visualise_meshgrid(A, B, y_grid, X, y):
    plt.contourf(A, B, y_grid, levels=np.linspace(y_grid.min(), y_grid.max(), 4), cmap=ListedColormap(['white', '#008B8B']))
    plt.title('Sample data')
    y_val_colours = ['blue' if y_val == 1 else '#CCCC00' for y_val in y]
    plt.scatter(X[:, 0], X[:, 1], c=y_val_colours)
    plt.show()

