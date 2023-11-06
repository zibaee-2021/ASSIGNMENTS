import numpy as np
import python_functions as py_func
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns

# if __name__ == '__main__':
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
    #
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
    # ds = np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1)

    # #### Naive Regression
    # a. Using polynomial basis for k=1 only i.e. $y = b$.
    #
    # MSEs_train_part_a, MSEs_test_part_a = py_func.split_dataset_and_compute_20_MSEs_with_ones(ds)
    # mean_MSE_train_part_a = np.mean(MSEs_train_part_a, axis=0)
    # mean_MSE_test_part_a = np.mean(MSEs_test_part_a, axis=0)
    # print(f'mean_MSE_train_part_a {mean_MSE_train_part_a}')
    # print(f'mean_MSE_test_part_a {mean_MSE_test_part_a}')
    # b. The constant function effectively determines the bias ($y$-intercept) of the linear regression.
    # It is the lowest predicted value of the dependent variable, the median house price.


    # c. For each of the 12 attributes, perform a linear regression using only the single attribute but incorporating
    # a bias term so that the inputs are augmented with an additional 1 entry, (xi , 1), so that we learn a weight
    # vector w ∈ R2.

    # (Does this mean average of the 12 weights or MSEs .. ?)

    # mean_for_each_of_12_attr_mse_train, mean_for_each_of_12_attr_mse_test = \
    #     py_func.split_dataset_and_compute_means_of_20_MSEs_with_single_attr(ds)
    # print(f'mean_MSE_train_part_c {mean_for_each_of_12_attr_mse_train}')
    # print(f'mean_MSE_test_part_c {mean_for_each_of_12_attr_mse_test}')

    # d. Perform linear regression using all of the data attributes at once.
    # Perform linear regression on the training set using this regressor, and incorporate a bias term as above.
    #
    # Calculate the MSE on the training and test sets and note down the results.
    # You should find that this method outperforms any of the individual regressors.

    # MSEs_train_part_d, MSEs_test_part_d = py_func.split_dataset_and_compute_means_of_20_MSEs_with_12_attrs(ds)
    # print(f'Mean MSE for train dataset, using all 12 attributes = {np.mean(MSEs_train_part_d)}')  # gives 25.3
    # print(f'Mean MSE for test dataset, using all 12 attributes = {np.mean(MSEs_test_part_d)}')  # gives 21.7


if __name__ == '__main__':
    # #### 1.3 Kernelised ridge regression
    # TRAIN
    ds = np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1)
    train_ds, test_ds = train_test_split(ds, test_size=1/3)
    mean_of_5folds_train, index_of_best_gamma_train, best_gamma_train, \
        index_of_best_sigma_train, best_sigma_train = \
        py_func.find_gamma_sigma_pair_with_lowest_MSE_using_gaussian_KRR(train_ds)
    print(f'index_of_best_gamma={index_of_best_gamma_train}, index_of_best_sigma={index_of_best_sigma_train}')

    # TRAIN HEATMAP
    np.savetxt("mean_of_5folds_train.csv", mean_of_5folds_train, delimiter=",")
    _, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(mean_of_5folds_train, annot=False, fmt=".2f", ax=ax)
    ax.set_xlabel('sigmas')
    ax.set_ylabel('gammas')
    lowest_MSE_train = np.full(mean_of_5folds_train.shape, np.nan)
    lowest_MSE_train[index_of_best_gamma_train, index_of_best_sigma_train] = \
        mean_of_5folds_train[index_of_best_gamma_train, index_of_best_sigma_train]
    sns.heatmap(lowest_MSE_train, annot=False, fmt=".05f", cmap='Reds', ax=ax, cbar=False, alpha=0.4)
    ax.set_title('Mean MSEs vs gammas & sigmas - train')
    plt.tight_layout()
    plt.savefig('plots/krr_mse_heatmap_train.jpg')
    plt.show()
    plt.close()

    # TRAIN LOG HEATMAP
    ln_mean_of_5folds_train = np.log(mean_of_5folds_train)
    np.savetxt("ln_of_mean_of_5folds_train.csv", ln_mean_of_5folds_train, delimiter=",")
    _, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(ln_mean_of_5folds_train, annot=False, fmt=".2f", ax=ax)
    ax.set_xlabel('sigmas')
    ax.set_ylabel('gammas')
    lowest_MSE_train = np.full(ln_mean_of_5folds_train.shape, np.nan)
    lowest_MSE_train[index_of_best_gamma_train, index_of_best_sigma_train] = \
        ln_mean_of_5folds_train[index_of_best_gamma_train, index_of_best_sigma_train]
    sns.heatmap(lowest_MSE_train, annot=False, fmt=".05f", cmap='Reds', ax=ax, cbar=False, alpha=0.4)
    ax.set_title('Ln mean MSEs vs gammas & sigmas - train')
    plt.tight_layout()
    plt.savefig('plots/krr_log_mse_heatmap_train.jpg')
    plt.show()

    # TEST --------------------------------------------------
    mean_of_5folds_test, index_of_best_gamma_test, best_gamma_test, \
        index_of_best_sigma_test, best_sigma_test = py_func.find_gamma_sigma_pair_with_lowest_MSE_using_gaussian_KRR(test_ds)
    print(f'index_of_best_gamma_test={index_of_best_gamma_test}, index_of_best_sigma_test={index_of_best_sigma_test}')

    # TEST HEATMAP
    np.savetxt("mean_of_5folds_test.csv", mean_of_5folds_test, delimiter=",")
    _, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(mean_of_5folds_test, annot=False, fmt=".2f", ax=ax)
    ax.set_xlabel('sigmas')
    ax.set_ylabel('gammas')
    lowest_MSE_test = np.full(mean_of_5folds_test.shape, np.nan)
    lowest_MSE_test[index_of_best_gamma_test, index_of_best_sigma_test] = \
        mean_of_5folds_test[index_of_best_gamma_test, index_of_best_sigma_test]
    sns.heatmap(lowest_MSE_test, annot=False, fmt=".05f", cmap='Reds', ax=ax, cbar=False, alpha=0.4)
    ax.set_title('Mean MSEs vs gammas & sigmas - test')
    plt.tight_layout()
    plt.savefig('plots/krr_mse_heatmap_test.jpg')
    plt.show()
    plt.close()

    # TEST (LOG HEATMAP)
    ln_mean_of_5folds_test = np.log(mean_of_5folds_test)
    np.savetxt("ln_of_mean_of_5folds_test.csv", ln_mean_of_5folds_test, delimiter=",")
    _, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(ln_mean_of_5folds_test, annot=False, fmt=".2f", ax=ax)
    ax.set_xlabel('sigmas')
    ax.set_ylabel('gammas')
    lowest_MSE_test = np.full(ln_mean_of_5folds_test.shape, np.nan)
    lowest_MSE_test[index_of_best_gamma_test, index_of_best_sigma_test] = \
        ln_mean_of_5folds_test[index_of_best_gamma_test, index_of_best_sigma_test]
    sns.heatmap(lowest_MSE_test, annot=False, fmt=".05f", cmap='Reds', ax=ax, cbar=False, alpha=0.4)
    ax.set_title('Ln mean MSEs vs gammas & sigmas - test')
    plt.tight_layout()
    plt.savefig('plots/krr_log_mse_heatmap_test.jpg')
    plt.show()



