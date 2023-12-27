import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import pandas as pd


def plot_mnist_digit(x):
    """
    Based on `plotChar[char_]` in poorCodeDemoDig.nb
    Plot the given 256 grayscale values to 16 x 16 image.
    :param x: NumPy array for one MNIST digit, containing 257 numbers.
    The first number is the MNIST digit (i.e. the label), the remaining 256 numbers are the flattened 16×16 greyscale
    pixels values corresponding to the digit.
    """
    assert len(x) == 257, 'Length of given array should be 257'
    data = np.array(x[1:257]).reshape((16, 16))
    plt.imshow(data, cmap='gray')
    plt.title(f"The number {x[0]}")
    plt.show()


def precompute_polykern_matrix(x1, x2, degree):
    ker_mat = x1 @ x2.T
    poly_ker_mat = ker_mat ** degree
    return poly_ker_mat


def _compute_polynomial_kernel(x1, x2, degree=3) -> float:
    """
    Based on `kerval[a_, b_]` in poorCodeDemoDig.nb: "function to compute kernel of a degree 3 polynomial".
    The computation is for two arrays, so might already be optimally vectorised via NumPy.
    :param x1: NumPy array of one MNIST digit, containing 256 numbers (i.e. flattened 16×16 greyscale pixels values).
    :param x2: As `x1`, but for a different MNIST digit.
    :param degree: Degree to use for the polynomial. (Three by default).
    """
    return (np.dot(x1, x2)) ** degree


def sign(sum_of_weighted_product) -> float:
    """
    Sign function
    :param sum_of_weighted_product:
    :return: -1.0 or +1.0
    """
    return -1.0 if sum_of_weighted_product <= 0.0 else 1.0


def predict_digit(x_upto_t, xt, t, alpha, d) -> float:
    """
    From poorCodeDemoDig.nb. "Computes the prediction of a classifier on a particular pattern" by taking the sum of
    the products of the class coefficients and kernel values, across all classes.
    :param x_upto_t: 2d array of training or test dataset, from data point t=0 up to previous data point, t-1.
    :param xt: 2d array of 256 numbers for current data point.
    :param t: The index of the current of the data point xt.
    :param alpha: Weight per data point for all classes and data points. 2d array (k_classes, m).
    :param d: degree for polynomial kernel.
    :return: Prediction of the class (i.e. digit) according to the given image pixels.
    """
    prediction = 0.0
    alpha_len = len(alpha)

    for i in range(t):
        xi = x_upto_t[i]
        poly_kernel_xixt = _compute_polynomial_kernel(x1=xi, x2=xt, degree=d)

        # convert indexing for alpha, because it keeps original length, regardless of num of epochs.
        if i < alpha_len:  # 7438
            i_for_alpha = i
        else:
            i_for_alpha = i % alpha_len

        prediction += alpha[i_for_alpha] * poly_kernel_xixt

    yt_hat = sign(prediction)
    return yt_hat


def predict_digit_vec(alpha_vec, kern_mat, k, m):
    """
    Compute summed product of alpha and precomputed kernel matrix for all values preceding, according to equation
    given in question sheet.
    :param alpha_vec: Weights per data point for every k class. Expected shape (k_classes, m).
    :param kern_mat: Precomputed kernel matrix. Expected shape (m, m).
    :param k: Number of classes to classify. Expected to be 10 for zipcombo.
    :param m: Number of data points. Expected to be 9298 * epochs for zipcombo.
    :return: summed weighted products of data points. Expected shape (k_classes, m).
    """
    assert alpha_vec.shape[0] == k
    assert alpha_vec.shape[1] == m
    assert kern_mat.shape[0] == m
    assert kern_mat.shape[1] == m
    preds = np.zeros((m, k))
    for t in range(1, m):
        a_slice = alpha_vec[:, :t]
        k_slice = kern_mat[:t, t].reshape(-1, 1)
        pred = np.dot(a_slice, k_slice)
        preds[t] = pred.T
    return preds.T


def multiply_by_epochs(epochs, y_vec, ds, alpha, k_mat):
    y_vec_ep = np.tile(A=y_vec, reps=epochs)
    ds_ep = np.tile(A=ds.T, reps=epochs)
    alpha_ep = np.tile(A=alpha, reps=epochs)
    k_mat_ep = np.tile(A=k_mat, reps=(epochs, epochs))
    return y_vec_ep, ds_ep.T, alpha_ep, k_mat_ep


def convert_y_to_vector(y, k_classes):
    """
    Convert digit values of y into +1 or -1 for all k classes. (Similar to one-hot encoding).
    :param y: 1d array of y values (labels). For zipcombo, values in set {0,1,2,3,4,5,6,7,8,9}
    :param k_classes: Number of classes to classify. (For zipcombo, expecting 10).
    :return: 2d array - ones, with shape (classes, num of data points). +1.0 to indicate class for data point.
    """
    m = len(y)
    y_vec = np.ones((k_classes, m))
    y_vec = -1 * y_vec
    m_indices = np.arange(m)
    k_indices = y.astype(int)
    y_vec[k_indices, m_indices] = 1.0
    return y_vec


def unused_code():
    # # These updates equation below (i.e. `alpha = alpha - sign(preds)`) was more effective at
    # # mis-classifying than classifying. DO NOT USE!
    # # Predictions for this data point
    # preds_at_t = preds[:, t]
    # # Convert raw values to -1 or 1
    # preds_signs_at_t = np.ones((k_classes, 1))
    # preds_signs_at_t[preds_at_t <= 0.0] = -1.0
    # current_alphas_for_misclassfd_at_t = alpha_vec[misclass_mask_at_t, t].reshape(-1, 1)
    # update_vals_for_misclassfd_at_t = preds_signs_at_t[misclass_mask_at_t].reshape(-1, 1)
    # updated_alphas_at_t = (current_alphas_for_misclassfd_at_t - update_vals_for_misclassfd_at_t).flatten()
    # alpha_vec[misclass_mask_at_t, t] = updated_alphas_at_t
    pass

if __name__ == '__main__':
    y = [1, 2, 4, 6, 7, 8]
    y = [1, 2]
    y = np.array(y)

    y_ = convert_y_to_vector(y, k_classes=10)
    print(y_)