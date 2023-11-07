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

# # PART 4a (with std devs added for part 5d)
# # Perform linear regression using a vector of ones only, producing a constant function.
# if __name__ == '__main__':
#     ds = np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1)
#     mean_mse_train, stdev_mse_train, mean_mse_test, stdev_mse_test = \
#         py_func.split_dataset_and_compute_mean_and_stdev_of_20_MSEs_with_ones(ds)
#     print(f'mean_mse_train {mean_mse_train}')
#     print(f'stdev_mse_train {stdev_mse_train}')
#     print(f'mean_mse_test {mean_mse_test}')
#     print(f'stdev_mse_test {stdev_mse_test}')
#     pass
# # PART 4c (with std devs added for part 5d)
# # Perform linear regression using each of the 12 attributes, one at a time.
# # if __name__ == '__main__':
#     ds = np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1)
#     _12_mse_means_train, _12_mse_stdevs_train, _12_mse_means_test, _12_mse_stdevs_test = \
#         py_func.split_dataset_and_compute_means_and_stdevs_of_20_MSEs_with_single_attr(ds)
#     # print(f'_12_mse_means_train {_12_mse_means_train}')
#     # print(f'_12_mse_stdevs_train {_12_mse_stdevs_train}')
#     # print(f'_12_mse_means_test {_12_mse_means_test}')
#     # print(f'_12_mse_stdevs_test {_12_mse_stdevs_test}')
#
#     _12_means_train_2dp = [round(num, 2) for num in _12_mse_means_train]
#     _12_stdevs_train_2dp = [round(num, 2) for num in _12_mse_stdevs_train]
#     _12_means_test_2dp = [round(num, 2) for num in _12_mse_means_test]
#     _12_stdevs_test_2dp = [round(num, 2) for num in _12_mse_stdevs_test]
#
#     col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','LSTAT']
#     table_latex = []
#     for i, colname in enumerate(col_names):
#         table_latex.append(f'LR ({colname})&{_12_means_train_2dp[i]}$\pm${_12_stdevs_train_2dp[i]}&'
#                            f'{_12_means_test_2dp[i]}$\pm${_12_stdevs_test_2dp[i]}\\\hline \n')
#     print(table_latex)
#     with open('latex/table5d.txt', 'w') as f:
#         f.write(''.join(table_latex))
#
#     pass

# PART 4d (with std devs added for part 5d)
# Perform linear regression using all of the 12 attributes together.
# if __name__ == '__main__':
#     ds = np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1)
#     mse_mean_of_all_12_attr_train, mse_stdev_of_all_12_attr_train, mse_mean_of_all_12_attr_test, mse_stdev_of_all_12_attr_test = \
#     py_func.split_dataset_and_compute_means_and_stdevs_of_20_MSEs_with_all_12_attrs(ds)
#     print(f'mse_mean_of_all_12_attr_train {mse_mean_of_all_12_attr_train}')
#     print(f'mse_stdev_of_all_12_attr_train {mse_stdev_of_all_12_attr_train}')
#     print(f'mse_mean_of_all_12_attr_test {mse_mean_of_all_12_attr_test}')
#     print(f'mse_stdev_of_all_12_attr_test {mse_stdev_of_all_12_attr_test}')
# if __name__ == '__main__':
#     # #### 1.3 Kernelised ridge regression
    # QUESTION 5 a-c
    # TRAIN: 5FOLD CV TO GET MSEs FOR ALL GAMMAS & SIGMAS. PRINT BEST GAMMA & SIGMA PAIR.
    # ds = np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1)
    # train_ds, test_ds = train_test_split(ds, test_size=1/3)
    # mean_mses_of_5folds_train, index_of_best_gamma_train, best_gamma_train, \
    #     index_of_best_sigma_train, best_sigma_train = \
    #     py_func.find_gamma_sigma_pair_with_lowest_MSE_using_gaussian_KRR(train_ds)
    # print(f'index_of_best_gamma_train={index_of_best_gamma_train}, '
    #       f'index_of_best_sigma_train={index_of_best_sigma_train}')

    # # TRAIN: GRID SEARCH HEATMAP OF MSEs HEATMAP
    # np.savetxt('grid_search/mean_mses_of_5folds_train.csv', mean_mses_of_5folds_train, delimiter=",")
    # _, ax = plt.subplots(figsize=(5, 5))
    # sns.heatmap(mean_mses_of_5folds_train, annot=False, fmt=".2f", ax=ax)
    # lowest_MSE_train = np.full(mean_mses_of_5folds_train.shape, np.nan)
    # lowest_MSE_train[index_of_best_gamma_train, index_of_best_sigma_train] = \
    #     mean_mses_of_5folds_train[index_of_best_gamma_train, index_of_best_sigma_train]
    # sns.heatmap(lowest_MSE_train, annot=False, fmt=".05f", cmap='Reds', ax=ax, cbar=False, alpha=0.4)
    # ax.set_title('Mean MSEs vs gammas & sigmas - train')
    # ax.set_xlabel('sigmas')
    # ax.set_ylabel('gammas')
    # plt.tight_layout()
    # plt.savefig('plots/krr_mse_heatmap_train.jpg')
    # plt.show()
    # # plt.close()

    # TRAIN: GRID SEARCH HEATMAP OF LN(MSEs)
    # ln_mean_mses_of_5folds_train = np.log(mean_mses_of_5folds_train)
    # np.savetxt("grid_search/ln_of_mean_mses_of_5folds_train.csv", ln_mean_mses_of_5folds_train, delimiter=",")
    # _, ax = plt.subplots(figsize=(5, 5))
    # sns.heatmap(ln_mean_mses_of_5folds_train, annot=False, fmt=".2f", ax=ax)
    # lowest_MSE_train = np.full(ln_mean_mses_of_5folds_train.shape, np.nan)
    # lowest_MSE_train[index_of_best_gamma_train, index_of_best_sigma_train] = \
    #     ln_mean_mses_of_5folds_train[index_of_best_gamma_train, index_of_best_sigma_train]
    # sns.heatmap(lowest_MSE_train, annot=False, fmt=".05f", cmap='Reds', ax=ax, cbar=False, alpha=0.4)
    # ax.set_title('Ln mean MSEs vs gammas & sigmas - train')
    # ax.set_xlabel('sigmas')
    # ax.set_ylabel('gammas')
    # plt.tight_layout()
    # plt.savefig('plots/krr_log_mse_heatmap_train.jpg')
    # plt.show()

    # USE BEST GAMMA & SIGMA TO COMPUTE MSEs FOR TRAIN & TEST (NO 5-FOLD CV):

    # GET BEST ALPHA_STARS (REGRESSION COEFFICIENT), USING BEST GAMMA & SIGMA, AND TRAIN DS:
#     X_train, y_train = train_ds[:, :12], train_ds[:, -1]
#     best_alpha_stars = py_func.solve_dual_optimisation(X_train=X_train, gamma=best_gamma_train,
#                                                        sigma=best_sigma_train, y_train=y_train)
#
#     # COMPUTE MSE FOR TRAIN DATASET USING ALPHA_STARS OF BEST GAMMA & SIGMA
#     sqrd_errors_train = []
#     mse_train = []
#     mse_train_using_best_gs = 0
#     X_train, y_train = train_ds[:, :12], train_ds[:, -1]
#     for i, (x_train_row, y_train_row) in enumerate(zip(X_train, y_train)):
#         y_train_pred = py_func.evaluation_of_regression(a_stars=best_alpha_stars, X_train=X_train,
#                                                         X_val_row=x_train_row, sigma=best_sigma_train)
#         sqrd_errors_train.append(np.square(y_train_pred - y_train_row))
#         mse_train_using_best_gs = np.mean(sqrd_errors_train)
#
#     print(f'mse_train_best_gs={mse_train_using_best_gs}')
#
#     # COMPUTE MSE FOR TEST DATASET USING ALPHA_STARS OF BEST GAMMA & SIGMA
#     sqrd_errors_test = []
#     mse_test = []
#     mse_test_using_best_gs = 0
#     X_test, y_test = test_ds[:, :12], test_ds[:, -1]
#     for i, (x_test_row, y_test_row) in enumerate(zip(X_test, y_test)):
#         y_test_pred = py_func.evaluation_of_regression(a_stars=best_alpha_stars, X_train=X_train,
#                                                        X_val_row=x_test_row, sigma=best_sigma_train)
#         sqrd_errors_test.append(np.square(y_test_pred - y_test_row))
#         mse_test_using_best_gs = np.mean(sqrd_errors_test)
#
#     print(f'mse_test_best_gs={mse_test_using_best_gs}')
#
if __name__ == '__main__':
    #### 1.3 Kernelised ridge regression
    # QUESTION 5d
    _20_MSEs_train, _20_MSEs_test = [], []
    ds = np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1)

    number_of_runs = 20
    for i in range(number_of_runs):
        print(f'{i+1}/20 KRR')
        # 1. Find best gamma and sigma using 5 fold CV:
        _, g_of_gs_pair_with_lowest_mse, _, s_of_gs_pair_with_lowest_mse, \
            _ = py_func.find_gamma_sigma_pair_with_lowest_MSE_using_gaussian_KRR(ds)

        sigma_powers = [7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13]
        gamma_powers = [-40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26]
        best_sigma = 2**sigma_powers[s_of_gs_pair_with_lowest_mse]
        best_gamma = 2**gamma_powers[g_of_gs_pair_with_lowest_mse]
        # 2. Split_dataset_2/3 to train, 1/3 test.
        train_ds, test_ds = train_test_split(ds, test_size=1 / 3, random_state=i)

        # 3. Perform krr with best gs-pair on train to compute coefficient (alpha_stars)
        X_train, y_train = train_ds[:, :12], train_ds[:, -1]
        best_alpha_stars = py_func.solve_dual_optimisation(X_train=X_train, gamma=best_gamma,
                                                           sigma=best_sigma, y_train=y_train)
        # 4. Predict train and test using alpha_stars coefficient, compute MSEs and append to list of 20.
        mse_train, mse_test = py_func.compute_krr_MSEs_for_train_and_test(a_stars_best=best_alpha_stars,
                                                                          best_sig=best_sigma, train_ds=train_ds,
                                                                          test_ds=test_ds)
        _20_MSEs_train.append(mse_train)
        _20_MSEs_test.append(mse_test)

    # 5. Calc mean and stddev of 20 MSEs for train and of 20 MSEs of test. Print/save values for table 2.
    krr_MSE_mean_train = np.mean(_20_MSEs_train)
    krr_MSE_stdev_train = np.std(_20_MSEs_train, ddof=1)
    krr_MSE_mean_test = np.mean(_20_MSEs_test)
    krr_MSE_stdev_test = np.std(_20_MSEs_test, ddof=1)
    with open('saved_values/krr_5d.txt', 'w') as f2:
        f2.write(f'krr_MSE_mean_train={krr_MSE_mean_train}\nkrr_MSE_stdev_train={krr_MSE_stdev_train}\n'
                 f'krr_MSE_mean_test={krr_MSE_mean_test}\nkrr_MSE_stdev_test {krr_MSE_stdev_test}\n')
    print(f'krr_MSE_mean_train={krr_MSE_mean_train}, krr_MSE_stdev_train={krr_MSE_stdev_train}')
    print(f'krr_MSE_mean_test={krr_MSE_mean_test}, krr_MSE_stdev_test={krr_MSE_stdev_test}')

    assert len(_20_MSEs_train) == number_of_runs, f'_20_MSEs_train is not expected length of {number_of_runs}'
    assert len(_20_MSEs_test) == number_of_runs, f'_20_MSEs_test is not expected length of {number_of_runs}'
