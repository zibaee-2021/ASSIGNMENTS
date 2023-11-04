import numpy as np
import python_functions as py_func

if __name__ == '__main__':
    # dataset_x, dataset_y = [1, 2, 3, 4], [3, 2, 0, 5]
    # X_k1_k2_k3_k4 = py_func.transform_dataset_by_polynom_basis_k1_to_k4(dataset_x)
    # weights_k1_k2_k3_k4 = py_func.compute_weights_of_lr_by_least_sqrs(X_k1_k2_k3_k4, y=np.array(dataset_y).reshape(-1, 1))
    # x_for_pred_and_plot = np.linspace(0, 5, 100)
    # y_preds = py_func.predict_with_polynom_func(weights_k1_k2_k3_k4, x_for_pred = x_for_pred_and_plot)
    # py_func.plot_polynoms_k1_k2_k3_k4(x=dataset_x, y=dataset_y, y_preds=y_preds, x_for_plot=x_for_pred_and_plot)

    # 1.1.1. (b) Give the equations corresponding to the curves fitted for k = 1, 2, 3.** <br>
    #
    # **Using the `weights_k1_k2_k3_k4` variable calculated in part (a), the equation is based on:**<br>

    # print(f'k1_x0 = {float(weights_k1_k2_k3_k4[0][0])}')
    # print(f'k2_x0 = {float(weights_k1_k2_k3_k4[1][0])}, k2_x1 = {float(weights_k1_k2_k3_k4[1][1])}')
    # print(f'k3_x0 = {float(weights_k1_k2_k3_k4[2][0])}, k3_x1 = {float(weights_k1_k2_k3_k4[2][1])},'
    #       f'k3_x2 = {float(weights_k1_k2_k3_k4[2][2])}')
    # print(f'k4_x0 = {float(weights_k1_k2_k3_k4[3][0])}, k4_x1 = {float(weights_k1_k2_k3_k4[3][1])},'
    #       f'k4_x2 = {float(weights_k1_k2_k3_k4[3][2])}, k4_x3 = {float(weights_k1_k2_k3_k4[3][3])}')

    # MSEs = py_func.calculate_MSEs(m=len(dataset_x), X=X_k1_k2_k3_k4, w=weights_k1_k2_k3_k4, y=dataset_y)
    # for i, MSE in zip([0, 1, 2, 3], MSEs):
    #     print(f'k={i + 1} MSE = {MSE}')

    # 1.1.2. (a) ii. Fit dataset to polynomial bases k=2, 5, 10, 14 and 18.
    # Superimpose their plots to (maybe) observe overfitting with increasing number of dimensions:

    # g_dataset_30x, g_dataset_30y = py_func.generate_dataset_about_g(num_of_data_pairs=30)
    # py_func.plot_g_0p07_and_sin_sqrd_2pi_x(x=g_dataset_30x, y=g_dataset_30y, x_for_plot=np.linspace(0, 1, 1000))
    #
    # X_k18_30 = py_func.transform_dataset_by_polynom_basis_k18(x=g_dataset_30x)
    # weights_k2_k5_k10_k14_k18 = py_func.compute_weights_of_lr_by_least_sqrs([X_k18_30[:, :2], X_k18_30[:, :5],
    #                                                                          X_k18_30[:, :10], X_k18_30[:, :14],
    #                                                                          X_k18_30], g_dataset_30y)
    # x_for_pred_and_plot = np.linspace(0, 1, 1000)
    # y_preds = py_func.predict_with_polynom_func(weights_k2_k5_k10_k14_k18, x_for_pred=x_for_pred_and_plot)
    # py_func.plot_polynom_k2_k5_k10_k14_k18(x=g_dataset_30x, y=g_dataset_30y, y_preds=y_preds,
    #                                        x_for_plot=x_for_pred_and_plot)


    # **1.1.2. (b) Calculate training set error of dataset S  $te_k(S)$ and plot its natural log against k to observe
    # reduction in training error with increasing dimensions.**

    # weights_k1_to_k18 will be used for making predictions in test set.
    # weights_k1_to_k18, train_errors_k1_to_k18 = py_func.compute_weights_and_train_errors_polynom()
    # py_func.plot_log_error_vs_k(k=list(range(1, 19)), log_error=np.log(train_errors_k1_to_k18))

    # 1.1.2. (c) Calculate test set error of (larger) dataset T   using model weights from linear regression of
    # training dataset.
    # Plot its natural log against k to observe how with increasing dimensions (leading to overfiting), the model is
    # essentially starting to fit the noise.

    # py_func.plot_log_error_vs_k(k=list(range(1, 19)),
    #                             log_error=np.log(py_func.compute_test_errors_polynom(w=weights_k1_to_k18)),
    #                             train_or_test='test')

    # weights_k1_to_k18, mean_train_errors_k1_to_k18_100runs = \
    #     py_func.train_weights_and_compute_mean_error_of_100runs_polynom()
    # py_func.plot_log_error_vs_k(k=list(range(1, 19)),
    #                             log_error=np.log(mean_train_errors_k1_to_k18_100runs), train_or_test='train')
    # mean_test_errors_k1_to_k18_100runs = py_func.compute_mean_error_of_100runs_test_polynom(w=weights_k1_to_k18)
    # py_func.plot_log_error_vs_k(k=list(range(1, 19)),
    #                             log_error=np.log(mean_test_errors_k1_to_k18_100runs), train_or_test='test')

    # 1.1.3. Repeat 2 (b)-(d) but use basis..
    # Hence, calculate training errors and plot against k, calculate test errors and plot against k, and repeat but
    # with average of 100 epochs.

    # weights_k1_to_k18_sin, train_errors_k1_to_k18_sin = py_func.compute_weights_and_train_errors_sine()
    # test_errors_k1_to_k18_sin = py_func.compute_test_errors_sine(w=weights_k1_to_k18_sin)
    # py_func.plot_log_errors_for_train_and_test_vs_k(k=list(range(1, 19)),
    #                                                 log_train_errors=np.log(train_errors_k1_to_k18_sin),
    #                                                 log_test_errors=np.log(test_errors_k1_to_k18_sin))

    # weights_k1_to_k18_sin, mean_train_errors_k1_to_k18_100runs = \
    #     py_func.train_weights_and_compute_mean_error_of_100runs_sine()
    # mean_test_errors_k1_to_k18_100runs = py_func.compute_mean_error_of_100runs_test_sine(w=weights_k1_to_k18_sin)
    # py_func.plot_log_errors_for_train_and_test_vs_k(k=list(range(1, 19)),
    #                                                 log_train_errors=np.log(mean_train_errors_k1_to_k18_100runs),
    #                                                 log_test_errors=np.log(mean_test_errors_k1_to_k18_100runs))

    ### 1.2 Filtered Boston housing and kernels

    # Downloaded csv in command line using:
    # `curl -o boston-filter.csv http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv`

    # dataset_np_headers_dropped
    ds = np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1)

    # #### Naive Regression
    # a. Using polynomial basis for k=1 only i.e. $y = b$.
    #
    # b. The constant function effectively determines the bias ($y$-intercept) of the linear regression.
    # It is the lowest predicted value of the dependent variable, the median house price.

    MSEs_train_part_a, MSEs_test_part_a = py_func.split_dataset_and_compute_20_MSEs_with_ones(ds)

    # c. For each of the 12 attributes, perform a linear regression using only the single attribute but incorporating
    # a bias term so that the inputs are augmented with an additional 1 entry, (xi , 1), so that we learn a weight
    # vector w ∈ R2.

    # (Does this mean average of the 12 weights or MSEs .. ?)

    MSEs_train_part_c, MSEs_test_part_c = py_func.split_dataset_and_compute_20_MSEs_with_single_attr(ds)

    means_train, means_test = [], []
    for MSEs_train_part_c_per_attr, MSEs_test_part_c_per_attr in zip(MSEs_train_part_c, MSEs_test_part_c):
        means_train.append(np.mean(MSEs_train_part_c_per_attr))
        means_test.append(np.mean(MSEs_test_part_c_per_attr))

    print(f'Means for each of the 12 attributes in train ds = \n{means_train}\n')
    print(f'Means for each of the 12 attributes in test ds = \n{means_test}')

    # d. Perform linear regression using all of the data attributes at once.
    # Perform linear regression on the training set using this regressor, and incorporate a bias term as above.
    #
    # Calculate the MSE on the training and test sets and note down the results.
    # You should find that this method outperforms any of the individual regressors.

    MSEs_train_part_d, MSEs_test_part_d = py_func.split_dataset_and_compute_20_MSEs_with_all_12_attr(ds)

    print(
        f'Mean MSE for train dataset, using all 12 attributes = {np.mean(MSEs_train_part_d)}')  # gives 65.399928735516
    print(f'Mean MSE for test dataset, using all 12 attributes = {np.mean(MSEs_test_part_d)}')  # gives 68.373652725872

    # #### 1.3 Kernelised ridge regression

    # A Kernel function is given as an element-wise product (I THOUGHT IT WAS DOT PRODUCT??:
    # $K_{i,j} = K(x_i, x_j)$
    # $l$ is the size of the training set, represented by `l` in the code.
    # Run all permutations of the regularisation hyperparameter `gamma` and the Gaussian kernel hyperparameter `sigma`.

    # Create a vector of gamma values [2^-40, 2^-39,...,2^-26]
    gammas = [2 ** pow for pow in list(range(-40, -25))]
    print(len(gammas))
    # Create vector of sigma values [2^7, 2^7.5, . . . , 2^12.5, 2^13]
    sigmas = []
    for pow in list(range(7, 14)):
        sigmas.append(2 ** pow)
        sigmas.append(2 ** (pow + 0.5))
    sigmas = sigmas[:-1]
    print(len(sigmas))

