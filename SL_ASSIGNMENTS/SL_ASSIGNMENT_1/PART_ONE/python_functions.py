import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold


def transform_dataset_by_polynom_basis_k1_to_k4(x: list) -> list:
    """
    Build the 4 polynomial function input values for the independent variable x, for each of the 4 basis variable from k=1 to k=4.
    :param x: The given independent variable values.
    :return: Four polynomial input values for k=1, k=2, k=3, k=4.
    """
    X_1 = np.ones((len(x), 1))  # k=1 gives x^0, so filled with 1s. This is the bias term (i.e. y-intercept).
    X_2 = np.array(x).reshape(-1, 1)  # k=2
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


def transform_dataset_by_polynom_basis_k18(x):
    """
    Build polynomial function input values for independent variable x, for every basis up to 18.
    (The other 4 basis vectors (k = 2, 5, 10 and 14) are then sliced from this).
    :param x: The given independent variable values.
    :return: Four polynomial input values for k=2, k=5, k=10, k=14, k=18.
    """
    X_k18 = np.column_stack((np.ones((len(x), 1)), x.reshape(-1, 1)))
    for k in range(3, 19):
        X_k18 = np.column_stack((X_k18, np.array([x_**(k - 1) for x_ in x]).reshape(-1, 1)))
    return X_k18


def plot_polynom_k2_k5_k10_k14_k18(x, y, y_preds, x_for_plot):
    _ , ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 1.5)
    ax.scatter(x, y, color='black', s=15)
    linewidth=0.5
    for y_pred, k in zip(y_preds, [2, 5, 10, 14, 18]):
        ax.plot(x_for_plot, y_pred, label=f'k={k}', linewidth=linewidth)
        linewidth += 0.5

    plt.xlabel('x')
    plt.ylabel('y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()


def compute_weights_and_train_errors_polynom() -> tuple:
    g_dataset_30x, g_dataset_30y = generate_dataset_about_g(num_of_data_pairs=30)
    X_k18_30 = transform_dataset_by_polynom_basis_k18(x=g_dataset_30x)
    X_k1_to_k18_30 = [X_k18_30[:, :i] for i in range(1, 19)]
    weights_k1_to_k18 = compute_weights_of_lr_by_least_sqrs(X=X_k1_to_k18_30, y=g_dataset_30y)
    train_errors_k1_to_k18 = calculate_MSEs(m=len(g_dataset_30x), X=X_k1_to_k18_30,
                                               w=weights_k1_to_k18, y=g_dataset_30y)
    return weights_k1_to_k18, train_errors_k1_to_k18


def plot_log_error_vs_k(k, log_error, train_or_test='train'):
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
    X_k1_to_k18_1000 = [X_k18_1000[:, :i] for i in range(1, 19)]
    return calculate_MSEs(m=len(g_dataset_1000x), X=X_k1_to_k18_1000, w=w, y=g_dataset_1000y)


def train_weights_and_compute_mean_error_of_100runs_polynom() -> tuple:
    train_errors_k1_to_k18_100runs = np.zeros((100, 18))
    weights_k1_to_k18, train_errors_k1_to_k18 = compute_weights_and_train_errors_polynom()
    train_errors_k1_to_k18_100runs[0] = train_errors_k1_to_k18
    for i in range(1, 100):
        _, train_errors_k1_to_k18 = compute_weights_and_train_errors_polynom()
        train_errors_k1_to_k18_100runs[i] = train_errors_k1_to_k18
    mean_train_errors_k1_to_k18 = np.mean(train_errors_k1_to_k18_100runs, axis=0)
    return weights_k1_to_k18, mean_train_errors_k1_to_k18


def compute_mean_error_of_100runs_test_polynom(w: list):
    test_errors_k1_to_k18_100runs = np.zeros((100, 18))
    for i in range(100):
        test_errors_k1_to_k18_100runs[i] = compute_test_errors_polynom(w)
    return np.mean(test_errors_k1_to_k18_100runs, axis=0)


def transform_dataset_by_sine_bases_k18(x):
    X_k18 = x.reshape(-1, 1)
    for k in range(2, 19):
        X_k18 = np.column_stack((X_k18, np.array([np.sin(k * np.pi * x_) for x_ in x]).reshape(-1, 1)))
    return X_k18


def compute_weights_and_train_errors_sine() -> tuple:
    g_dataset_30x, g_dataset_30y = generate_dataset_about_g(num_of_data_pairs=30)
    X_k18_30 = transform_dataset_by_sine_bases_k18(x=g_dataset_30x)
    X_k1_to_k18_30 = [X_k18_30[:, :i] for i in range(1, 19)]
    weights_k1_to_k18 = compute_weights_of_lr_by_least_sqrs(X=X_k1_to_k18_30, y=g_dataset_30y)
    train_errors_k1_to_k18 = calculate_MSEs(m=len(g_dataset_30x), X=X_k1_to_k18_30,
                                               w=weights_k1_to_k18, y=g_dataset_30y)
    return weights_k1_to_k18, train_errors_k1_to_k18


def plot_log_errors_for_train_and_test_vs_k(k, log_train_errors, log_test_errors):
    _ , ax = plt.subplots()
    ax.set_xlim(1, 18)
    ax.scatter(k, log_train_errors, label='train', color='blue', s=5)
    ax.scatter(k, log_test_errors, label='test', color='red', s=5)
    ax.plot(k, log_train_errors, color='lightblue')
    ax.plot(k, log_test_errors, color='salmon')
    plt.xlabel('k')
    plt.ylabel(f'natural log of error')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()


def compute_test_errors_sine(w: list) -> list:
    g_dataset_1000x, g_dataset_1000y = generate_dataset_about_g(num_of_data_pairs=1000)
    X_k18_1000 = transform_dataset_by_sine_bases_k18(x=g_dataset_1000x)
    X_k1_to_k18_1000 = [X_k18_1000[:, :i] for i in range(1, 19)]
    test_errors_k1_to_k18 = calculate_MSEs(m=len(g_dataset_1000x), X=X_k1_to_k18_1000, w=w, y=g_dataset_1000y)
    return test_errors_k1_to_k18


def train_weights_and_compute_mean_error_of_100runs_sine() -> tuple:
    train_errors_k1_to_k18_100runs = np.zeros((100, 18))
    weights_k1_to_k18, train_errors_k1_to_k18 = compute_weights_and_train_errors_sine()
    train_errors_k1_to_k18_100runs[0] = train_errors_k1_to_k18
    for i in range(1, 100):
        _, train_errors_k1_to_k18 = compute_weights_and_train_errors_sine()
        train_errors_k1_to_k18_100runs[i] = train_errors_k1_to_k18
    mean_train_errors_k1_to_k18 = np.mean(train_errors_k1_to_k18_100runs, axis=0)
    return weights_k1_to_k18, mean_train_errors_k1_to_k18


def compute_mean_error_of_100runs_test_sine(w: list):
    test_errors_k1_to_k18_100runs = np.zeros((100, 18))
    for i in range(100):
        test_errors_k1_to_k18_100runs[i] = compute_test_errors_sine(w)
    return np.mean(test_errors_k1_to_k18_100runs, axis=0)


def fit_lr_and_calculate_mse(m_train: int, x_train, y_train, m_test: int, x_test, y_test) -> tuple:
    weights = compute_weights_of_lr_by_least_sqrs(X=[x_train], y=y_train)
    mse_train = calculate_MSEs(m=m_train, X=[x_train], w=weights, y=y_train)
    mse_test = calculate_MSEs(m=m_test, X=[x_test], w=weights, y=y_test)
    return mse_train[0], mse_test[0]


def split_dataset_and_compute_mean_and_stdev_of_20_MSEs_with_ones(ds) -> tuple:
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
    assert (len(_20_mse_train) == 20), 'length of _20_mse_train is not expected size of 20'
    assert (len(_20_mse_test) == 20), 'length of _20_mse_test is not expected size of 20'
    mean_mse_train, stdev_mse_train = np.mean(_20_mse_train), np.std(_20_mse_train, ddof=1)
    mean_mse_test, stdev_mse_test = np.mean(_20_mse_test), np.std(_20_mse_test, ddof=1)
    return mean_mse_train, stdev_mse_train, mean_mse_test, stdev_mse_test


def split_dataset_and_compute_means_and_stdevs_of_20_MSEs_with_single_attr(ds) -> tuple:
    _12_mse_means_train, _12_mse_stdevs_train = [], []
    _12_mse_means_test, _12_mse_stdevs_test = [], []
    _20_mse_train, _20_mse_test = [], []

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
        assert (len(_20_mse_train) == 20), 'length of _20_mse_train is not expected size of 20'
        assert (len(_20_mse_test) == 20), 'length of _20_mse_test is not expected size of 20'
        _12_mse_means_train.append(np.mean(_20_mse_train))
        _12_mse_stdevs_train.append(np.std(_20_mse_train, ddof=1))
        _12_mse_means_test.append(np.mean(_20_mse_test))
        _12_mse_stdevs_test.append(np.std(_20_mse_test, ddof=1))
    assert (len(_12_mse_means_train) == 12), 'length of _12_mse_means_train is not expected size of 12'
    assert (len(_12_mse_stdevs_train) == 12), 'length of _12_mse_stdevs_train is not expected size of 12'
    assert (len(_12_mse_means_test) == 12), 'length of _12_mse_means_test is not expected size of 12'
    assert (len(_12_mse_stdevs_test) == 12), 'length of _12_mse_stdevs_test is not expected size of 12'
    return _12_mse_means_train, _12_mse_stdevs_train, _12_mse_means_test, _12_mse_stdevs_test


def get_x_train_y_train_x_test_y_test(m_train: int, train_ds, m_test: int, test_ds) -> tuple:
    X_train_all_attr = train_ds[:, 0: 12]
    ones_train = np.ones((m_train, 1))
    X_train = np.column_stack((ones_train, X_train_all_attr))
    y_train = train_ds[:, -1]
    X_test_all_attr = test_ds[:, 0: 12]
    ones_test = np.ones((m_test, 1))
    X_test = np.column_stack((ones_test, X_test_all_attr))
    y_test = test_ds[:, -1]
    return X_train, y_train, X_test, y_test


def split_dataset_and_compute_means_and_stdevs_of_20_MSEs_with_all_12_attrs(ds) -> tuple:
    _20_mse_train, _20_mse_test = [], []

    for i in range(20):  # serves dual purpose: loop 20 times and provide seed for unique splits.
        train_dataset, test_dataset = train_test_split(ds, test_size=1 / 3, random_state=i)
        m_train, m_test = train_dataset.shape[0], test_dataset.shape[0]
        X_train, y_train, X_test, y_test = get_x_train_y_train_x_test_y_test(m_train=m_train, train_ds=train_dataset,
                                                                             m_test=m_test, test_ds=test_dataset)
        mse_train, mse_test = fit_lr_and_calculate_mse(m_train=m_train, x_train=X_train, y_train=y_train,
                                                       m_test=m_test, x_test=X_test, y_test=y_test)
        _20_mse_train.append(mse_train)
        _20_mse_test.append(mse_test)
    assert (len(_20_mse_train) == 20), 'length of _20_mse_train is not expected size of 20'
    assert (len(_20_mse_test) == 20), 'length of _20_mse_test is not expected size of 20'
    mse_mean_of_all_12_attr_train, mse_stdev_of_all_12_attr_train = np.mean(_20_mse_train), np.std(_20_mse_train,
                                                                                                   ddof=1)
    mse_mean_of_all_12_attr_test, mse_stdev_of_all_12_attr_test = np.mean(_20_mse_test), np.std(_20_mse_test, ddof=1)
    return mse_mean_of_all_12_attr_train, mse_stdev_of_all_12_attr_train, mse_mean_of_all_12_attr_test, mse_stdev_of_all_12_attr_test


# PYTHON FUNCTIONS FOR GAUSSIAN KERNEL RIDGE REGRESSION:

def gaussian_kernel(X, sig: float):
    """
    Generates the Gaussian kernel matrix for the given input matrix and sigma value.
    :param X: Input matrix. Expected to the training dataset.
    :param sig: Bandwidth (i.e. variance) parameter for the Gaussian kernel.
    :return: Kernel matrix.
    """
    num_of_rows_of_x = X.shape[0]
    kernel_matrix = np.empty((num_of_rows_of_x, num_of_rows_of_x))
    for i in range(num_of_rows_of_x):
        for j in range(num_of_rows_of_x):
            pairwise_diff = X[i] - X[j]
            sqrd_norm = np.square(np.linalg.norm(pairwise_diff))
            kernel_matrix[i][j] = np.exp(-1 * sqrd_norm / (2 * np.square(sig)))
    return kernel_matrix


def gaussian_kernel_vectorised(X, sig):
    """
    Generates the Gaussian kernel matrix for the given input matrix and sigma value.
    :param X: Input matrix. Expected to the training dataset.
    :param sig: Bandwidth (i.e. variance) parameter for the Gaussian kernel.
    :return: Kernel matrix.
    """
    sqrd_distances = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=-1)
    kernel_matrix = np.exp(-1 * sqrd_distances / (2 * np.square(sig)))
    return kernel_matrix


def evaluation_of_regression(a_stars, X_train, X_val_row, sigma) -> float:
    """
    Apply regression function by multiplying each regression coefficient (`alpha_star`) for each training
    example, with kernel matrix of that training example and a fix test (i.e. validation) example, and taking the
    sum for all examples, as per equation 13 in question sheet.
    :param a_stars: The regression coefficients (`alpha_stars`), one for each training data point.
    :param X_train: Training dataset.
    :param X_val_row: One example of the validation dataset (i.e. a row of the feature dataset).
    :param sigma: Bandwidth of kernel function.
    :return: Predicted values of dependent variable y. Calculated for each row of X_val, and mean is returned.
    """
    # Vectorised rather than loop.
    pairwise_diff = X_train - X_val_row
    sqrd_norm = np.square(np.linalg.norm(pairwise_diff, axis=1))
    kernel = np.exp(-1 * sqrd_norm / (2 * np.square(sigma)))
    y_preds = a_stars * kernel
    y_pred_for_given_x_val_row = float(np.sum(y_preds))
    return y_pred_for_given_x_val_row


def solve_dual_optimisation(X_train, gamma, sigma, y_train):
    """
    Calculate optimal regression coefficient(s),`alpha_star`, of dual optimisation formula according to equation
    12 of question sheet. This solves the kernel ridge regression.
    :param X_train: The training data features of dataset.
    :param gamma: Regularisation parameter.
    :param sigma: Parameter for Gaussian kernel.
    :param y_train: The training data labels in column vector format.
    :return: The regression coefficients, one for each training example (row of 12 numbers). e.g. shape (404,)
    """
    # kernel_matrix = gaussian_kernel(X_train, sigma)
    kernel_matrix = gaussian_kernel_vectorised(X=X_train, sig=sigma)
    l = X_train.shape[0]
    I = np.identity(l)
    right = (gamma * l * I)
    alpha_stars = (np.linalg.inv(kernel_matrix + right)) @ y_train.T
    return alpha_stars


def generate_gammas_and_sigmas() -> tuple:
    """
    Create gammas values [2^-40, 2^-39, . . . , 2^-26].
    Create sigma values [2^7, 2^7.5, . . . , 2^12.5, 2^13].
    :return:
    """
    gammas = [2 ** pow for pow in list(range(-40, -25))]
    sigmas = []
    for pow in list(range(7, 14)):
        sigmas.append(2 ** pow)
        sigmas.append(2 ** (pow + 0.5))
    sigmas = sigmas[:-1]
    return gammas, sigmas


def generate_5_folds_from_dataset(ds) -> tuple:
    kf = KFold(n_splits=5, shuffle=True)
    _5_X_train, _5_y_train, _5_X_valid, _5_y_valid = [], [], [], []
    for train_i, valid_i in kf.split(ds):
        X_train, X_valid = ds[train_i][:, :12], ds[valid_i][:, :12]
        y_train, y_valid = ds[train_i][:, -1], ds[valid_i][:, -1]
        _5_X_train.append(X_train)
        _5_X_valid.append(X_valid)
        _5_y_train.append(y_train)
        _5_y_valid.append(y_valid)
    return _5_X_train, _5_y_train, _5_X_valid, _5_y_valid


def find_gamma_sigma_pair_with_lowest_MSE_using_gaussian_KRR(ds) -> tuple:
    """
    # 1. Loop through each of 5 train-validation dataset folds.
    # 2. For each fold, loop through every combination of gamma and sigma (gs-pair).
    # 3. For each gs-pair, loop through each example (row) of X_val to predict the corresponding y.
    # 4. Calculate squared error (i.e. difference between predicted y and corresponding y_val).
    # 5. Calculate mean of squared errors for this validation fold.
    # 6. For each fold, store each mean MSE for every gs-pairs (195 pairs), producing a list of 5 x 195 MSE values.
    # 7. Take the mean of each gs-pair across the 5 folds and pull out the gs-pair with the lowest MSE.
    :param ds: Expected to be 2/3 of full dataset (Boston-filter.csv). Shape (337, 13).
    :return: The gamma, sigma pair that gives the lowest MSE across 5 CV folds.
    """
    _5_X_train, _5_y_train, _5_X_valid, _5_y_valid = generate_5_folds_from_dataset(ds)
    gammas, sigmas = generate_gammas_and_sigmas()
    mean_of_sqrd_errors_per_gs_pair_for_one_x_val = np.zeros((len(gammas), len(sigmas)))
    MSEs_for_each_gs_pair_for_all_5folds = []

    # 1. Loop through each of 5 train-validation dataset folds.
    for X_train, y_train, X_val, y_val in zip(_5_X_train, _5_y_train, _5_X_valid, _5_y_valid):
        # 2. For each fold, loop through each combination of gamma and sigma (gs-pair).
        for g, gamma in enumerate(gammas):
            for s, sigma in enumerate(sigmas):
                a_stars_for_this_gs_pair_and_fold = solve_dual_optimisation(X_train=X_train, gamma=gamma,
                                                                            sigma=sigma, y_train=y_train)
                # 3. For each gs-pair, loop through each example (row) of X_val to predict the corresponding y.
                sqrd_errors = []
                for i, (x_val_row, y_val_row) in enumerate(zip(X_val, y_val)):
                    y_val_pred = evaluation_of_regression(a_stars=a_stars_for_this_gs_pair_and_fold,
                                                           X_train=X_train, X_val_row=x_val_row, sigma=sigma)

                    # 4. Calculate squared error (i.e. difference between predicted y and corresponding y_val).
                    sqrd_errors.append(np.square(y_val_pred - y_val_row))

                assert len(sqrd_errors) == len(y_val), 'sqrd_errors list is not the expected length'
                # 5. Calculate mean of squared errors for this validation set.
                mean_of_sqrd_errors = np.mean(sqrd_errors)
                mean_of_sqrd_errors_per_gs_pair_for_one_x_val[g][s] = mean_of_sqrd_errors
        # 6. For each fold, store each mean MSE for every gs-pairs (195 pairs), producing a list of 5 x 195 MSE values.
        MSEs_for_each_gs_pair_for_all_5folds.append(mean_of_sqrd_errors_per_gs_pair_for_one_x_val)
        mean_of_sqrd_errors_per_gs_pair_for_one_x_val = np.zeros((len(gammas), len(sigmas)))

    fold1 = MSEs_for_each_gs_pair_for_all_5folds[0]
    fold2 = MSEs_for_each_gs_pair_for_all_5folds[1]
    fold3 = MSEs_for_each_gs_pair_for_all_5folds[2]
    fold4 = MSEs_for_each_gs_pair_for_all_5folds[3]
    fold5 = MSEs_for_each_gs_pair_for_all_5folds[4]

    assert len(MSEs_for_each_gs_pair_for_all_5folds) == 5
    # 7. Take the mean of each gs-pair across the 5 folds and pull out the gs-pair with the lowest MSE.
    mean_mse_of_5folds = (fold1 + fold2 + fold3 + fold4 + fold5) / 5
    # np.argmin() flattens 2d to 1d array, so unravel_index() is required afterwards to restore original shape.
    index_of_lowest_mse = np.argmin(mean_mse_of_5folds)
    g_of_gs_pair_with_lowest_mse, s_of_gs_pair_with_lowest_mse = np.unravel_index(index_of_lowest_mse, mean_mse_of_5folds.shape)
    return mean_mse_of_5folds, g_of_gs_pair_with_lowest_mse, gammas[g_of_gs_pair_with_lowest_mse], \
        s_of_gs_pair_with_lowest_mse, sigmas[s_of_gs_pair_with_lowest_mse]


def compute_krr_MSEs_for_train_and_test(a_stars_best, best_sig, train_ds, test_ds):
    X_train, y_train = train_ds[:, :12], train_ds[:, -1]
    X_test, y_test = test_ds[:, :12], test_ds[:, -1]
    sqrd_errors_train, sqrd_errors_test = [], []

    for i, (x_train_row, y_train_row, x_test_row, y_test_row) in enumerate(zip(X_train, y_train, X_test, y_test)):
        y_train_pred = evaluation_of_regression(a_stars=a_stars_best, X_train=X_train,
                                                X_val_row=x_train_row, sigma=best_sig)
        y_test_pred = evaluation_of_regression(a_stars=a_stars_best, X_train=X_train,
                                               X_val_row=x_test_row, sigma=best_sig)
        sqrd_errors_train.append(np.square(y_train_pred - y_train_row))
        sqrd_errors_test.append(np.square(y_test_pred - y_test_row))
    mse_train = np.mean(sqrd_errors_train)
    mse_test = np.mean(sqrd_errors_test)

    return mse_train, mse_test