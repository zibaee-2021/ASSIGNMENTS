import numpy as np


def _compute_polynomial_kernel(x1, x2, degree=3):
    """
    Based on `kerval[a_, b_]` in poorCodeDemoDig.nb: "function to compute kernel of a degree 3 polynomial".
    The computation is for two arrays, so might already be optimally vectorised via NumPy.
    :param x1: NumPy array of one MNIST digit, containing 256 numbers (i.e. flattened 16×16 greyscale pixels values).
    :param x2: As `x1`, but for a different MNIST digit.
    :param degree: Degree to use for the polynomial. (Three by default).
    """
    return (np.dot(x1, x2)) ** degree


def _sign(sum_of_weighted_product) -> float:
    """
    Sign function
    :param sum_of_weighted_product:
    :return: -1.0 or +1.0
    """
    return -1.0 if sum_of_weighted_product <= 0.0 else 1.0


def predict_digit(x_upto_t, xt, t, alpha_upto_t, d) -> float:
    """
    As `classpredk[dat_, pat_, cl_]` from poorCodeDemoDig.nb. "Computes the prediction of a classifier on a
    particular pattern" by taking the sum of the products of the class coefficients and kernal values, across all
    classes.
    :param x_upto_t: NumPy array of training or test dataset, from data point t=0 up to current data point, t-1.
    :param xt: NumPy array of 256 numbers representing the flattened 16x16 greyscale pixels values
    for one example digit.
    :param t: The current iteration of data point, xt.
    :param alpha_upto_t: NumPy array of weight per data point, from data point t=0 up to current data point, t-1
    :return: Prediction of the class (i.e. digit) according to the given image pixels.
    """
    prediction = 0.0

    for i in range(t):
        xi = x_upto_t[i]
        poly_kernal = _compute_polynomial_kernel(x1=xi, x2=xt, degree=d)
        prediction += alpha_upto_t[i] * poly_kernal

    yt_hat = _sign(prediction)
    return yt_hat
