import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


def transform_dataset_by_polynom_basis_k1_to_k4(x: list) -> list:
    """
    Build the 4 polynomial function input values for the independent variable x, for each of the 4 basis variable from k=1 to k=4.
    :param x: The given independent variable values.
    :return: Four polynomial input values for k=1, k=2, k=3, k=4.
    """
    X_1 = np.ones((len(x), 1))  # k=1 gives x^0, so filled with 1s. This is the bias term (i.e. y-intercept).
    X_2 = np.array(x).reshape(-1, 1) # k=2
    X_2 = np.column_stack((X_1, X_2))
    X_3 = np.array([x_**2 for x_ in x]).reshape(-1, 1) # k=3
    X_3 = np.column_stack((X_2, X_3))
    X_4 = np.array([x_**3 for x_ in x]).reshape(-1, 1) # k=4
    X_4 = np.column_stack((X_3, X_4))
    return [X_1, X_2, X_3, X_4]


def compute_weights_of_lr_by_least_sqrs(X: list, y, use_pinv=True) -> list:
    """
    Calculate coefficients of linear regression using least squares of given data set.
    This is also referred to as 'training the model'.
    :param X: Independent variable datasets. List of NumPy arrays.
    :param y: Dependent variable values. NumPy array.
    :param use_pinv: True to use (Moore-Penrose) pseudo-inverse matrix instead of explicitly
                     using (X^t*X)^-1 *X^t*y. True by default.
    :return: Coefficients (aka weights).
    """
    weights = []
    for X_ in X:
        if use_pinv:
            w = np.linalg.pinv(X_)@y
        else:
            X_t = X_.T
            w = np.linalg.inv(X_t @ X_) @ X_t @ y
        weights.append(w.flatten())
    return weights


def predict_with_polynom_func(w: list, x_for_pred) -> list:
    """
    Evaluate polynomial for range of given independent variable values to calculate predicted values of
    dependent variable using given coefficients.
    Note: NumPy's poly1d() expects x^n + .. + x^1 + x^0, so I need to reverse the order of w_T
    :param w: Weights for polynomial function.
    :param x_for_pred: Independent variable values to use for predicting dependent variable values. NumPy array.
    :return: Predicted values of dependent variable.
    """
    y_preds = []
    for w_ in w:
        w_T_for_np = np.flip(w_.T).flatten()
        y_pred = np.poly1d(w_T_for_np)(x_for_pred)
        y_preds.append(y_pred)
    return y_preds


def plot_polynoms_k1_k2_k3_k4(x, y, y_preds: list, x_for_plot) -> None:
    """
    :param x: The independent variable (input), NumPy array.
    :param y: The dependent variable (aka label), NumPy array.
    :param y_preds: Predicted values of y for each of the four k values, NumPy array.
    """
    _ , ax = plt.subplots(facecolor='white')
    ax.set_xlim(0, 5)
    ax.set_ylim(-4, 8)
    ax.scatter(x, y, color='red')
    for y_pred, k in zip(y_preds, [1,2,3,4]):
        ax.plot(x_for_plot, y_pred, label=f'k={k}')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()


def calculate_MSEs(m: int, X: list, w: list, y: list) -> list:
    """
    Calculate mean squared error for each of the different linear regression model learned by least squares
    for the polynomial functions of different bases.
    :param m: Number of samples in dataset (expecting either 4, 30, or 1000).
    :param X: List of NumPy arrays. Each array contains transformed datasets of independent variable values,
              according to the basis. Each array is 2-dimensional i.e. (4,1), not (4,).
    :param w: Weights learned by least squares linear regression. List of NumPy arrays.
    Each array is 1-dimensional, hence (4,), not (4,1).
    A dot product of a 2-d array and 1-d array like X[1] @ w[1] (i.e. (4,2) @ (2,) produces a 1-d array of (4,).
    :param y: The dependent variable values corresponding to the independent variables of the dataset.
    As a python list, addition or subtraction is compatible with a 1-d NumPy array.
    So X[1] @ w[1] - y works and produces a 1-d NumPy array. So (4,2) @ (2,) - python list of length 4 gives (4,).
    :return: The mean squared error.
    """
    MSEs = []

    for i in range(len(X)):
        MSEs.append((1 / m) * (X[i] @ w[i] - y).T @ (X[i] @ w[i] - y))
    return MSEs


def sin_sqrd_2pi_x(x: float) -> float:
    """
    Apply sine(2*pi*x)^2 to given input number.
    :param x: Input number
    :return: Output number, a function of input.
    """
    return np.square(np.sin(2 * np.pi * x))


def g_0p07(x: float) -> float:
    """
    Apply sine(2*pi*x)^2 with random normally-distributed noise to given input number,
    using standard deviation of 0.07.
    :param x: Input number.
    :return: Output number, a function of input.
    """
    mean, std_dev, num_of_samples = 0, 0.07, 1
    epsilon = np.random.normal(loc=mean, scale=std_dev, size=num_of_samples)
    epsilon = float(epsilon[0])
    return sin_sqrd_2pi_x(x) + epsilon


def generate_dataset_about_g(num_of_data_pairs: int) -> tuple:
    """
    Generate dataset of given number of independent variables sampled uniformly from
    interval [0, 1] and corresponding dependent variables
    :return: Dataset S, a NumPy array and list.
    """
    x = np.random.uniform(0, 1, num_of_data_pairs)
    y = [g_0p07(x_) for x_ in x]
    return x, y


def plot_g_0p07_and_sin_sqrd_2pi_x(x, y, x_for_plot):
    _ , ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.scatter(x, y, color='red', s=10)
    y_func=[sin_sqrd_2pi_x(x_) for x_ in x_for_plot]
    ax.plot(x_for_plot, y_func, linewidth=1)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def transform_dataset_by_polynom_basis_k18(x: list) -> list:
    """
    Build polynomial function input values for independent variable x, for every basis up to 18.
    (The other 4 basis vectors (k = 2, 5, 10 and 14) are then sliced from this).
    :param x: The given independent variable values.
    :return: Four polynomial input values for k=2, k=5, k=10, k=14, k=18.
    """
    X_k18 = np.column_stack((np.ones((len(x), 1)), np.array(x).reshape(-1, 1)))
    for k in range(3, 19):
        X_k18 = np.column_stack((X_k18, np.array([x_**(k - 1) for x_ in x]).reshape(-1, 1)))
    return X_k18


def plot_polynom_k2_k5_k10_k14_k18(x, y, y_preds, x_for_plot):
    _ , ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 1.5)
    ax.scatter(x, y, color='red', s=10)
    linewidth=0.5
    for y_pred, k in zip(y_preds, [2,5,10,14,18]):
        ax.plot(x_for_plot, y_pred, label=f'k={k}', linewidth=linewidth)
        linewidth += 0.5

    plt.xlabel('x')
    plt.ylabel('y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()


def compute_training_errors_polynom() -> tuple:
    g_dataset_30x, g_dataset_30y = generate_dataset_about_g(num_of_data_pairs=30)
    X_k18_30 = transform_dataset_by_polynom_basis_k18(x=g_dataset_30x)
    X_k1_to_k18_30 = [X_k18_30[:,:i] for i in range(1, 19)]
    weights_k1_to_k18 = compute_weights_of_lr_by_least_sqrs(X=X_k1_to_k18_30, y=g_dataset_30y)
    training_errors_k1_to_k18 = calculate_MSEs(m=len(g_dataset_30x), X=X_k1_to_k18_30,
                                               w=weights_k1_to_k18, y=g_dataset_30y)
    return weights_k1_to_k18, training_errors_k1_to_k18


def plot_log_error_vs_k(k, log_error, train_or_test='training'):
    _ , ax = plt.subplots()
    ax.set_xlim(1, 18)
    # ax.set_ylim(-5,0)
    ax.scatter(k, log_error, color='red', s=10)
    ax.plot(k, log_error)
    plt.xlabel('k')
    plt.ylabel(f'natural log of {train_or_test} error')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def compute_test_errors_polynom(w: list) -> list:
    g_dataset_1000x, g_dataset_1000y = generate_dataset_about_g(num_of_data_pairs=1000)
    X_k18_1000 = transform_dataset_by_polynom_basis_k18(x=g_dataset_1000x)
    X_k1_to_k18_1000 = [X_k18_1000[:,:i] for i in range(1, 19)]
    return calculate_MSEs(m=len(g_dataset_1000x), X=X_k1_to_k18_1000, w=w, y=g_dataset_1000y)


def run_training_polynom_100_times() -> list:
    training_errors_k1_to_k18_list = []
    weights_k1_to_k18_list = []
    for _ in range(100):
        weights_k1_to_k18, training_errors_k1_to_k18 = compute_training_errors_polynom()
        training_errors_k1_to_k18_list.append(training_errors_k1_to_k18)
        weights_k1_to_k18_list.append(weights_k1_to_k18)
    # weights_k1_to_k18 = np.mean(weights_k1_to_k18_list, axis=0)
    return training_errors_k1_to_k18_list


def run_test_polynom_100_times(w: list) -> list:
    test_errors_k1_to_k18_list = []
    for _ in range(100):
        test_errors_k1_to_k18 = compute_test_errors_polynom(w)
        test_errors_k1_to_k18_list.append(test_errors_k1_to_k18)
    return test_errors_k1_to_k18_list


def transform_dataset_by_sine_bases_k18(x: list) -> list:
    X_k1_to_k18_list = []
    X_k1_to_k18_list.append(np.array([np.sin(1 * np.pi * x_) for x_ in x]).reshape(-1, 1))
    for k in range(2, 19):
        # temp_ary = np.array([np.sin(k * np.pi * x_) for x_ in x]).reshape(-1,1)
        # X_k1_to_k18_list.append(temp_ary)
        X_k1_to_k18_list.append(np.array([np.sin(k * np.pi * x_) for x_ in x]).reshape(-1,1))
    return X_k1_to_k18_list


def compute_training_errors_sine() -> tuple:
    g_dataset_30x, g_dataset_30y = generate_dataset_about_g(num_of_data_pairs=30)
    X_k1_to_k18_30 = transform_dataset_by_sine_bases_k18(x=g_dataset_30x)
    weights_k1_to_k18 = compute_weights_of_lr_by_least_sqrs(X=X_k1_to_k18_30, y=g_dataset_30y)
    training_errors_k1_to_k18 = calculate_MSEs(m=len(g_dataset_30x), X=X_k1_to_k18_30,
                                               w=weights_k1_to_k18, y=g_dataset_30y)
    return weights_k1_to_k18, training_errors_k1_to_k18


def compute_test_errors_sine(w: list) -> list:
    g_dataset_1000x, g_dataset_1000y = generate_dataset_about_g(num_of_data_pairs=1000)
    X_k1_to_k18_1000 = transform_dataset_by_sine_bases_k18(x=g_dataset_1000x)
    test_errors_k1_to_k18 = calculate_MSEs(m=len(g_dataset_1000x), X=X_k1_to_k18_1000, w=w, y=g_dataset_1000y)
    return test_errors_k1_to_k18


def run_training_sine_100_times() -> list:
    training_errors_k1_to_k18_list = []
    weights_k1_to_k18_list = []
    for _ in range(100):
        weights_k1_to_k18, training_errors_k1_to_k18 = compute_training_errors_sine()
        training_errors_k1_to_k18_list.append(training_errors_k1_to_k18)
        # weights_k1_to_k18_list.append(weights_k1_to_k18)
        # weights_k1_to_k18 = np.mean(weights_k1_to_k18_list, axis=0) # heterogenous array size,
        # so would need more complicated logic to compute means of weights.
    return training_errors_k1_to_k18_list


def run_test_sine_100_times(w: list)->list:
    test_errors_k1_to_k18_list = []
    for _ in range(100):
        test_errors_k1_to_k18 = compute_test_errors_sine(w)
        test_errors_k1_to_k18_list.append(test_errors_k1_to_k18)
    return test_errors_k1_to_k18_list


def fit_lr_and_calculate_mse(m_train: int, x_train, y_train, m_test: int, x_test, y_test) -> tuple:
    weights = compute_weights_of_lr_by_least_sqrs(X=[x_train], y=y_train)
    # Reusing polynomial predictor function from part 1.1 but only up to degree 0 or 1,
    # hence basis for k=1 or k=2, i.e. y = b or y = mx + b.
    mse_train = calculate_MSEs(m=m_train, X=[x_train], w=weights, y=y_train)
    mse_test = calculate_MSEs(m=m_test, X=[x_test], w=weights, y=y_test)
    return mse_train[0], mse_test[0]


def split_dataset_and_compute_20_MSEs_with_ones(ds) -> tuple:
    _20_mse_train = []
    _20_mse_test = []

    for i in range(20):  # serves dual purpose: loop 20 times and provide seed for unique splits.

        train_dataset, test_dataset = train_test_split(ds, test_size=1 / 3, random_state=i)
        m_train = train_dataset.shape[0]
        m_test = test_dataset.shape[0]

        x_train = np.ones((m_train, 1))
        y_train = train_dataset[:, -1]

        x_test = np.ones((m_test, 1))
        y_test = test_dataset[:, -1]

        mse_train, mse_test = fit_lr_and_calculate_mse(m_train=m_train, x_train=x_train, y_train=y_train,
                                                       m_test=m_test, x_test=x_test, y_test=y_test)
        _20_mse_train.append(mse_train)
        _20_mse_test.append(mse_test)

    return _20_mse_train, _20_mse_test


def split_dataset_and_compute_20_MSEs_with_single_attr(ds) -> tuple:
    each_of_12_attr_mse_train = []
    each_of_12_attr_mse_test = []

    for col_num in range(ds.shape[1] - 1):
        _20_mse_train = []
        _20_mse_test = []
        for i in range(20):  # serves dual purpose: loop 20 times and provide seed for unique splits.

            train_dataset, test_dataset = train_test_split(ds, test_size=1 / 3, random_state=i)

            m_train = train_dataset.shape[0]
            x_train_single_attr = train_dataset[:, col_num]
            x_train_single_attr = x_train_single_attr.reshape(-1, 1)
            ones_train = np.ones((m_train, 1))
            x_train = np.column_stack((ones_train, x_train_single_attr))
            y_train = train_dataset[:, -1]

            m_test = test_dataset.shape[0]
            x_test_single_attr = test_dataset[:, col_num]
            x_test_single_attr = x_test_single_attr.reshape(-1, 1)
            ones_test = np.ones((m_test, 1))
            x_test = np.column_stack((ones_test, x_test_single_attr))
            y_test = test_dataset[:, -1]

            mse_train, mse_test = fit_lr_and_calculate_mse(m_train=m_train, x_train=x_train, y_train=y_train,
                                                                   m_test=m_test, x_test=x_test, y_test=y_test)
            _20_mse_train.append(mse_train)
            _20_mse_test.append(mse_test)

        each_of_12_attr_mse_train.append(_20_mse_train)
        each_of_12_attr_mse_test.append(_20_mse_test)

    return each_of_12_attr_mse_train, each_of_12_attr_mse_test


if __name__ == '__main__':
    # dataset = np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1)
    #
    a, b = split_dataset_and_compute_20_MSEs_with_ones()
    # a, b = split_dataset_and_compute_20_MSEs_with_single_attr(np.genfromtxt('boston-filter.csv', delimiter=',', skip_header=1))

    # dataset_x, dataset_y = [1, 2, 3, 4], [3, 2, 0, 5]
    # X_k1_k2_k3_k4 = transform_dataset_by_polynom_basis_k1_to_k4(dataset_x)
    # weights_k1_k2_k3_k4 = compute_weights_of_lr_by_least_sqrs(X_k1_k2_k3_k4, y=np.array(dataset_y).reshape(-1, 1))
    # calculate_MSEs(m=len(dataset_x), X=X_k1_k2_k3_k4, w=weights_k1_k2_k3_k4, y=dataset_y)

    # compute_training_errors_polynom()
    print(b)
