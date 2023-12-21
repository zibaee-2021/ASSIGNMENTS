import python_functions_part2 as pf2
import numpy as np
from matplotlib import pyplot as plt


# if __name__ == '__main__':
#     # Question 6:
#     X, y = pf2.generate_random_h()
#     num_of_grid_points = 100
#     A, B, y_grid_empty = pf2.generate_equidistant_coords_in_2d_grid(num_of_grid_points=num_of_grid_points)
#     y_grid = pf2.perform_knn(v=3, A=A, B=B, X=X, y=y, num_of_grid_points=num_of_grid_points,
#                              y_grid=y_grid_empty)
#     pf2.visualise_meshgrid(y_grid, X, y)


# if __name__ == '__main__':
#     # Question 7:
#     num_of_centres = 5000
#     y_errors = []
#     all_100_gen_errors = np.zeros((100, 50))
#
#     for i in range(100):
#         print(f'{i+1}/100 ********')
#         num_of_grid_points = 100
#         # "Sample a h from pH"
#         X, y = pf2.generate_noisy_data_with_biased_coin(num_of_centres=num_of_centres)
#         # "Build a k-NN model with 4000 training points sampled from ph(x,y)."
#         # As far as I can tell there's nothing to do for "building a k-NN model" other than choosing k.. ?
#
#         # "Run k-NN estimate generalisation error (for this run) using 1000 test points sampled from ph(x,y)."
#         # This seems to be just comparing the randomly-generated y-labels in test set with what the k-neighbouring
#         # labels in the existing training set would have classified it as. It's odd.
#         X_train, X_test = X[:4000, :], X[4000:, :]
#         y_train, y_test = y[:4000], y[4000:]
#
#         X_test_ = X_test[:, np.newaxis, :]
#         diffs = X_test_ - X_train
#         dists = np.sqrt(np.sum(np.square(diffs), axis=2))
#         v = 1
#         indices_of_closest_v = np.argpartition(dists, v, axis=1)[:, :v]
#         maj_class = pf2.get_maj_class_of_neighbrs_vectorised(indices_of_closest_v, y)
#         gen_errors = np.where(maj_class != y_test, 1, 0)
#         gen_errors = np.sum(gen_errors) / 1000
#         all_100_gen_errors[v, 0] = gen_errors
#         # Note index 0 will be empty as I'm using v (1-49) as index.
#
#         for v in range(2, 50):
#             print(f'k={v} ----------')
#             # Reshape (1000,2) to (1000,1,2) to enable broadcasting
#             X_test_ = X_test[:, np.newaxis, :]
#             diffs = X_test_ - X_train
#             dists = np.sqrt(np.sum(np.square(diffs), axis=2))
#             indices_of_closest_v = np.argpartition(dists, v, axis=1)[:, :v]
#             maj_class = pf2.get_maj_class_of_neighbrs_vectorised(indices_of_closest_v, y)
#             gen_errors = np.where(maj_class != y_test, 1, 0)
#             gen_errors = np.sum(gen_errors) / 1000
#             all_100_gen_errors[i, v] = gen_errors
#
#         all_100_gen_errors = np.vstack(all_100_gen_errors)
#
#     mean_gen_errors = np.mean(all_100_gen_errors, axis=0)
#     print(mean_gen_errors)
#     np.savetxt('saved/mean_gen_errors.txt', mean_gen_errors, delimiter=',', fmt='%d')


if __name__ == '__main__':
    with open('saved/mean_gen_errors_printout.txt', 'r') as f:
        mean_gen_errors_vs_v = f.read()

    mean_gen_errors_vs_v = mean_gen_errors_vs_v.split()
    mean_gen_errors_vs_v = [float(num) for num in mean_gen_errors_vs_v]
    _1_49 = list(range(1, 50))
    _1_49 = np.array(_1_49)
    plt.scatter(x=[list(range(1, 50))], y=mean_gen_errors_vs_v)
    plt.plot(_1_49, mean_gen_errors_vs_v)
    plt.show()