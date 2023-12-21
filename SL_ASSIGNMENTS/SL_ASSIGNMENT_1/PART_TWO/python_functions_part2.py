import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


# QUESTION 7 FUNCTIONS: ------------------------------------------------------------
def get_maj_class_of_neighbrs_vectorised(indices_of_closest_v, y) -> int:
    closest_v_classes = y[indices_of_closest_v]
    row_sums = np.sum(closest_v_classes, axis=1)
    # Rule: the majority is 1 if the sum is >= n/2, else it's 0
    majority_values = (row_sums >= (closest_v_classes.shape[1] / 2)).astype(int)
    # Reshape to get an array of shape (1000, 1)
    majority_values = majority_values.reshape(-1, 1)
    return majority_values


def _calculate_distances_between_Xheads(Xheads0, Xheads1, S, num, i_=None ):
    """
    Calculate Euclidean distance between the given X heads coin flip and every other data point in the dataset.
    :param Xheads0: A single data point, which is one axis of the X heads data point.
    :param Xheads1: A single data point, which is the other axis of the X heads data point.
    :param S: Dataset. 2D numpy array, e.g. shape (100,2) for 100 centres.
    :param num_data_centres: Number of data points in the 2d hypothesis space.
    :return: Distances from given grid coordinates to each centre in given dataset.
    """
    distances = np.zeros(num)
    for i, X_centre in enumerate(S):
        if i is not None:
            if i == i_:
                continue
        distances[i] = np.sqrt((Xheads0 - X_centre[0]) ** 2 + (Xheads1 - X_centre[1]) ** 2)
    return distances


def _perform_knn_for_heads(v, i, X_heads, X, y, num_of_centres) -> int:
    dists = _calculate_distances_between_Xheads(Xheads0=X_heads[0], Xheads1=X_heads[1], S=X, num=num_of_centres, i_=i)
    indices_of_closest_v = _get_closest_neighbours(dists=dists, v=v)
    majority_class = _get_majority_class_of_neighbours(indices_of_closest_v, y=y)
    return majority_class


# Step 4: Assign these majority values back to corresponding positions in ar1
def generate_noisy_data_with_biased_coin(num_of_centres=5000):
    """
    Generate noisy data, by sampling an x in same way as for generate_random_h(), and the corresponding y value
    generated by flipping biased coin P(heads) = 0.8/P(tails) = 0.2. If heads comes up then y = h_{S,3}(x) otherwise y
    is sampled uniformly at random from {0, 1}, as in generate_random_h()
    :param num_of_centres:
    :return:
    """
    X, y = generate_random_h(num_of_samples=5000, prob_of_success=0.8)
    for i, (y_, x_) in enumerate(zip(y, X)):
        if y_ == 1:
            y[i] = _perform_knn_for_heads(v=3, i=i, X_heads=x_, X=X, y=y, num_of_centres=num_of_centres)
    return X, y


def generate_noisy_data_with_biased_coin_vectorised(num_of_centres=5000):
    X, y = generate_random_h(num_of_samples=num_of_centres, prob_of_success=0.8)
    mask = y == 1
    y_masked = y[mask]
    # Expand ary2 to (5000, 1, 2) and (1, 5000, 2) to compute pairwise differences
    diffs = dists_masked[:, np.newaxis, :] - ary2[np.newaxis, :, :]
    # Compute the L2 norms. This will result in a (5000, 5000) shaped array
    l2_norms = np.sqrt(np.sum(diffs ** 2, axis=-1))

    # We don't need the diagonal elements which are the distances of the rows with themselves
    # Set them to a large number so they don't affect the argpartition
    np.fill_diagonal(l2_norms, np.inf)

    dists[i] = np.sqrt((Xheads0 - X_centre[0]) ** 2 + (Xheads1 - X_centre[1]) ** 2)
    indices_of_closest_v = _get_closest_neighbours(dists=dists, v=3)
    majority_class = _get_majority_class_of_neighbours(indices_of_closest_v, y=y)




# QUESTION 6 FUNCTIONS: ------------------------------------------------------------
def visualise_meshgrid(y_grid, X, y):
    A, B = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    plt.contourf(A, B, y_grid, levels=np.linspace(y_grid.min(), y_grid.max(), 4), cmap=ListedColormap(['white', '#008B8B']))
    plt.title('Sample data')
    y_val_colours = ['blue' if y_val == 1 else '#CCCC00' for y_val in y]
    plt.scatter(X[:, 0], X[:, 1], c=y_val_colours)
    plt.savefig('plots/3nn_contour_grid_Q6.jpg')
    plt.show()


def _get_majority_class_of_neighbours(indices_of_closest_v, y) -> int:
    closest_v_classes = y[indices_of_closest_v]
    counts_of_0_1 = np.bincount(closest_v_classes)
    majority_class = np.argmax(counts_of_0_1)
    return majority_class


def _get_closest_neighbours(dists, v):
    indices_of_lowest3 = np.argsort(dists)[:v]
    return indices_of_lowest3


def _calculate_distances_between(point0, point1, S, num_of_centres):
    """
    Calculate Euclidean distance between a point on the grid (akin to coordinates) and every data point in the dataset.
    :param point0: A single data point, from one array of np.meshgrid() points.
    :param point1: A single data point, from one array of np.meshgrid() points.
    :param S: Dataset. 2D numpy array, e.g. shape (100,2) for 100 centres.
    :param num_of_centres: Number of data points in the 2d hypothesis space.
    :return: Distances from given grid coordinates to each centre in given dataset.
    """
    distances = np.zeros(num_of_centres)
    for i, X_centre in enumerate(S):
        distances[i] = np.sqrt((point0 - X_centre[0]) ** 2 + (point1 - X_centre[1]) ** 2)
    return distances


def perform_knn(v, A, B, X, y, num_of_grid_points, y_grid, num_of_centres):
    for i in range(num_of_grid_points):
        for j in range(num_of_grid_points):
            dists = _calculate_distances_between(point0=A[i][j], point1=B[i][j], S=X,
                                                 num_of_centres=num_of_centres)
            indices_of_closest_v = _get_closest_neighbours(dists=dists, v=v)
            majority_class = _get_majority_class_of_neighbours(indices_of_closest_v, y=y)
            y_grid[i][j] = majority_class
    return y_grid


def generate_equidistant_coords_in_2d_grid(num_of_grid_points):
    a, b = np.linspace(0, 1, num_of_grid_points), np.linspace(0, 1, num_of_grid_points)
    A, B = np.meshgrid(a, b)
    y_grid_empty = np.zeros((num_of_grid_points, num_of_grid_points))
    return A, B, y_grid_empty


def generate_random_h(num_of_samples=100, num_of_trials=1, prob_of_success=0.5) -> tuple:
    """
    Generate data (i.e. centres) uniformly at random from [0, 1]^2 with the corresponding labels sampled uniformly at
    random from {0, 1}.
    :param num_of_samples: Number of centres.
    :param num_of_trials:
    :param prob_of_success:
    :return:
    """
    X = np.random.uniform(low=0, high=1, size=(num_of_samples, 2))
    y = np.random.binomial(n=num_of_trials, p=prob_of_success, size=num_of_samples)
    return X, y
