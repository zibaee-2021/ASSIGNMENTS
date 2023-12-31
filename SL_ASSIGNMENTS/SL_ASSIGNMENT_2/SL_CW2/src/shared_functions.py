import numpy as np
from numba import jit
from scipy.spatial.distance import cdist


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


def compute_polykern_matrix(x1, x2, degree):
    """
    Compute the polynomial kernel matrix with two datasets given each as 2d arrays.
    :param x1: One 2d array.
    :param x2: The other 2d array.
    :param degree: Degree to use for polynomial kernel.
    :return: The polynomial kernel matrix.
    """
    ker_mat = np.dot(x1, x2.T)
    poly_ker_mat = ker_mat ** degree
    return poly_ker_mat


def compute_gauss_kern_matrix(x1, x2, c):
    """
    Compute the Gaussian kernel matrix with two datasets given each as 2d arrays.
    :param x1: One 2d array.
    :param x2: The other 2d array.
    :param c: Width parameter to use for the Gaussian kernel equation. (Equivalent to 1/sigma^2).
    :return: The Gaussian kernel matrix.
    """
    sqrd_euclid_norm = cdist(x1, x2, 'sqeuclidean')
    Gauss_ker_mat = np.exp(-c * sqrd_euclid_norm)
    return Gauss_ker_mat


def predict_with_trained_alpha(trained_alpha, k_mat, y):
    """
    Compute predictions with trained weights and kernel matrix. Count mistakes compared to true labels.
    :param trained_alpha: Trained weights.
    :param k_mat: Kernel matrix.
    :param y: Labels (epoch-tiled). 1d array.
    :return: Number of mistakes made in prediction.
    """
    # FOR EACH EPOCH, USE TRAINED WEIGHTS `alpha_vec` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    predictions_with_trained_alpha = np.dot(trained_alpha, k_mat)
    # digit & index happen to be same number for 10 classes with 0-based indexing.
    predicted_y = np.argmax(predictions_with_trained_alpha, axis=0).reshape(-1, 1)
    true_y = y.reshape(-1, 1).astype(np.int32)
    mistakes = predicted_y != true_y
    mistakes = np.sum(mistakes)
    return mistakes


def test_kp(ds, k_mat, trained_alpha, y, d_or_c):
    """
    Perform predictions on test dataset split and calculate numbers of mis-classifications using given kernel matrix
    and pre-trained weights from corresponding training dataset split. (All given values are epoch-tiled).
    :param k_mat: Pre-computed kernel matrix (epoch-tiled) for corresponding training dataset split and test set split.
    :param trained_alpha: Trained weights (epoch-tiled), trained on corresponding training dataset split.
    :param y: Labels (epoch-tiled). 1d array.
    :param d_or_c: Degree to use for polynomial kernel or width to use for Gaussian kernel evaluation.
    :return: The error rate percentage for given dataset split.
    """
    # USE TRAINED WEIGHTS `trained_alpha` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    k_mat = k_mat.T  # non-symmetric kernel matrix needs correct orientation for dot product with trained_alpha
    mistakes = predict_with_trained_alpha(trained_alpha, k_mat, y)
    print(f'Number of test mistakes for d or c {d_or_c} = {mistakes}')
    error_rate_prct = 100 * (mistakes / len(ds))
    print(f'Test error for d or c {d_or_c} = {error_rate_prct} %')
    return error_rate_prct


def train_kp(y, k_mat, d_or_c, k_classes: int):
    """
    Perform a form of online learning by updating the weight (`alpha`) by updating its value the weighted kernel
    evaluation whenever its prediction is wrong.
    :param y: Labels for training dataset split of MNIST. As 1d array with shape (m, 257) where `m` is number of
    datapoints (epoch-tiled).
    :param k_mat: Kernel matrix for given training dataset (epoch-tiled).
    :param d_or_c: Degree used for polynomial kernel or width used for Gaussian kernel evaluation.
    :param k_classes: Number of classes for which to perform classification.
    :return: The error rate percentage and the trained weights for the given dataset, width, k_classes and epochs.
    """
    y_vec = convert_y_to_vector(y, k_classes)
    m = len(k_mat)
    alpha_vec = np.zeros((k_classes, m))
    assert alpha_vec.shape[0] == y_vec.shape[0] == k_classes
    assert k_mat.shape[0] == k_mat.shape[1] == alpha_vec.shape[1] == y_vec.shape[1]

    preds = np.zeros((k_classes, m))

    # ITERATE THROUGH ALL DATA POINTS:
    for t in range(m):
        a_slice = alpha_vec[:, :t]
        k_slice = k_mat[t, :t].reshape(-1, 1)
        # MAKE PREDICTIONS FOR THIS DATA POINT ACROSS ALL CLASSES:
        pred = np.dot(a_slice, k_slice)
        # UPDATE WEIGHTS FOR ANY THAT MISCLASSIFIED THIS DATA POINT (IF PRODUCT OF TRUE Y AND PREDICTED IS <= 0)
        preds[:, t] = pred.reshape(-1)
        prod_y_and_preds_at_t = y_vec[:, t] * preds[:, t]
        # STORE WHICH DIGITS ARE MISCLASSIFIED IN BOOLEAN MASK:
        misclass_mask_at_t = prod_y_and_preds_at_t <= 0
        y_vec_with_which_to_update_alpha = y_vec[misclass_mask_at_t, t].reshape(-1)
        alpha_vec[misclass_mask_at_t, t] = y_vec_with_which_to_update_alpha

        # ADDING HERE A POSSIBLE LEARNING RATE HYPERPARAMETER:
        # learning_rate = 0.2
        # alpha_vec[misclass_mask_at_t, t] += learning_rate * y_vec[misclass_mask_at_t, t].reshape(-1)

    # AFTER ALL EPOCHS, USE TRAINED WEIGHTS `alpha_vec` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    mistakes = predict_with_trained_alpha(alpha_vec, k_mat, y)
    print(f'Number of training mistakes {mistakes}')

    error_rate_prct = 100 * (mistakes / m)
    print(f'Train error for d or c {d_or_c} = {error_rate_prct} %')
    return error_rate_prct, alpha_vec


def calc_mean_error_per_deg_by_5f_cv(ds, mean_val_error_per_degree, _80split, epochs, degree, k_classes):
    # GET DATA POINTS AND LABELS FROM TRAINING SET SPLIT:
    x_train_80 = _80split[:, 1:]
    y_train_80 = _80split[:, 0]

    # CALC INDICES FOR 5-FOLDS:
    k_folds = 5
    folds = np.arange(0, _80split.shape[0], _80split.shape[0] // k_folds)
    folds[-1] = ds.shape[0] - 1
    start_end_k_folds = list(zip(folds[:-1], folds[1:]))

    error_prct_kfolds = np.zeros(k_folds)
    # MAKE 5-FOLD CV DATASETS:
    for i, (k_start, k_end) in enumerate(start_end_k_folds):
        x_train_cv = np.concatenate((x_train_80[:k_start], x_train_80[k_end:]), axis=0)
        y_train_cv = np.concatenate((y_train_80[:k_start], y_train_80[k_end:]))
        x_val_cv = x_train_80[k_start: k_end]
        y_val_cv = y_train_80[k_start: k_end]

        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_train_cv = compute_polykern_matrix(x1=x_train_cv, x2=x_train_cv, degree=degree)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_cv, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_cv, reps=(epochs, epochs))

        # TRAIN WEIGHTS (not interested in error rate from training fold):
        _, trained_alpha = train_kp(y=y, k_mat=k_mat, d_or_c=degree, k_classes=k_classes)

        # USE TRAINED WEIGHTS TO RUN *VALIDATION* TESTS. RECORD ERROR FOR THIS DEGREE:
        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_val = compute_polykern_matrix(x1=x_train_cv, x2=x_val_cv, degree=degree)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_val = kernel_matrix_val.T
        k_mat = np.tile(A=kernel_matrix_val, reps=epochs)

        # VALIDATION TEST WITH TRAINED WEIGHTS & CALC ERRORS FOR THIS DEGREE:
        error_prct_kfolds[i] = test_kp(ds=ds, k_mat=k_mat, trained_alpha=trained_alpha, y=y_val_cv, d_or_c=degree)
    # AFTER ALL 5 FOLDS, CALC MEANS OF VALIDATION TESTS FOR THIS DEGREE:
    # RECORD MEAN VALIDATION ERROR PER DEGREE
    mean_val_error_per_degree[degree] = np.mean(error_prct_kfolds)

    return mean_val_error_per_degree
