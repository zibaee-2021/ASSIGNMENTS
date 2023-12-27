import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import os


def _convert_y_to_vector(y, k_classes):
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


def _compute_polykern_matrix(x1, x2, degree):
    """
    Compute the polynomial kernel matrix with two datasets.
    :param x1:
    :param x2:
    :param degree:
    :return:
    """
    ker_mat = np.dot(x1, x2.T)
    poly_ker_mat = ker_mat ** degree
    return poly_ker_mat


def _predict_with_trained_alpha(trained_alpha, k_mat, y):
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
    # predicted_y = np.argmax(predictions_with_trained_alpha, axis=0).reshape(1, -1)
    predicted_y = np.argmax(predictions_with_trained_alpha, axis=0).reshape(-1, 1)
    # predicted_y = np.argmax(predictions_with_trained_alpha, axis=0)
    # predicted_y = predicted_y.reshape(-1, 1)
    true_y = y.reshape(-1, 1).astype(np.int32)
    mistakes = predicted_y != true_y
    mistakes = np.sum(mistakes)
    return mistakes


def test_kp(k_mat, trained_alpha, y, degree):
    """
    Perform predictions on test dataset split and calculate numbers of misclassifications using given kernel matrix
    and pre-trained weights from corresponding training dataset split. (All given values are epoch-tiled).
    :param k_mat: Pre-computed kernel matrix (epoch-tiled) for corresponding training dataset split and test set split.
    :param trained_alpha: Trained weights (epoch-tiled), trained on corresponding training dataset split.
    :param y: Labels (epoch-tiled). 1d array.
    :param degree: Degree to use for polynomial kernel evaluation.
    :return: The error rate percentage for given dataset split.
    """
    # assert k_mat.shape[0] == trained_alpha.shape[1]

    # USE TRAINED WEIGHTS `trained_alpha` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    k_mat = k_mat.T  # non-symmetric kernel matrix needs correct orientation for dot product with trained_alpha
    if k_mat.shape[0] == 240:
        pass
    mistakes = _predict_with_trained_alpha(trained_alpha, k_mat, y)
    print(f'Number of test mistakes for degree {degree} = {mistakes}')
    error_rate_prct = mistakes / len(ds)
    error_rate_prct *= 100
    print(f'Test error for degree {degree} = {error_rate_prct} %')
    return error_rate_prct


def train_kp(y, k_mat, degree: int, k_classes: int):
    """
    Perform a form of online learning by updating the weight (`alpha`) by updating its value the weighted kernel
    evaluation whenever its prediction is wrong.
    :param y: Labels for training dataset split of MNIST. As 1d array with shape (m, 257) where `m` is number of
    datapoints (epoch-tiled).
    :param k_mat: Kernel matrix for given training dataset (epoch-tiled).
    :param degree: Degree to use for polynomial kernel evaluation.
    :param k_classes: Number of classes for which to perform classification.
    :return: The error rate percentage and the trained weights for the given dataset, degree, k_classes and epochs.
    """
    y_vec = _convert_y_to_vector(y, k_classes)
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
        # print(f"Type of y_vec[misclass_mask_at_t, t]: {type(y_vec[misclass_mask_at_t, t])}")
        # print(f"Shape of y_vec[misclass_mask_at_t, t]: {y_vec[misclass_mask_at_t, t].shape}")

        learning_rate = 1
        y_vec_with_which_to_update_alpha = learning_rate * y_vec[misclass_mask_at_t, t].reshape(-1)
        alpha_vec[misclass_mask_at_t, t] = y_vec_with_which_to_update_alpha

    if k_mat.shape[0] == 240:
        a = 400
    # AFTER ALL EPOCHS, USE TRAINED WEIGHTS `alpha_vec` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    mistakes = _predict_with_trained_alpha(alpha_vec, k_mat, y)
    print(f'Number of training mistakes {mistakes}')

    error_rate_prct = 100 * (mistakes / m)
    print(f'Train error for degree {degree} = {error_rate_prct} %')
    return error_rate_prct, alpha_vec


def _calc_mean_error_per_deg_by_5f_cv(mean_val_error_per_degree, _80split, epochs, degree, k_classes):
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
        kernel_matrix_train_cv = _compute_polykern_matrix(x1=x_train_cv, x2=x_train_cv, degree=degree)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_cv, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_cv, reps=(epochs, epochs))

        if k_mat.shape[0] == 240:
            a = 100
        # TRAIN WEIGHTS (not interested in error rate from training fold):
        _, trained_alpha = train_kp(y=y, k_mat=k_mat, degree=degree, k_classes=k_classes)

        # USE TRAINED WEIGHTS TO RUN *VALIDATION* TESTS. RECORD ERROR FOR THIS DEGREE:
        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_val = _compute_polykern_matrix(x1=x_train_cv, x2=x_val_cv, degree=degree)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_val = kernel_matrix_val.T
        k_mat = np.tile(A=kernel_matrix_val, reps=epochs)

        if k_mat.shape[1] == 240:
            a = 200

        # VALIDATION TEST WITH TRAINED WEIGHTS & CALC ERRORS FOR THIS DEGREE:
        error_prct_kfolds[i] = test_kp(k_mat=k_mat, trained_alpha=trained_alpha, y=y_val_cv, degree=degree)
    # AFTER ALL 5 FOLDS, CALC MEANS OF VALIDATION TESTS FOR THIS DEGREE:
    # RECORD MEAN VALIDATION ERROR PER DEGREE
    mean_val_error_per_degree[degree] = np.mean(error_prct_kfolds)

    return mean_val_error_per_degree


# ---------- QUESTION 2. Cross validation to find best d ------------------------------------------------------------


def run_cv_kp(ds, degrees, num_of_runs=20, k_classes=10, epochs=3, write_results=False):
    """
    Train weights (`alpha`) on 80% dataset splits for given number of epochs. Run test with trained weights on
    corresponding 20% dataset split. Compute error rates +/- std dev for each split for both training and test split
    averaged over given number of independent runs.
    :param ds: Dataset.
    :param degrees: Degrees to try with polynomial kernel evaluation. (Expected to be within range 1 and 7)
    :param num_of_runs: Number of independent runs. 20 by default.
    :param k_classes: Number of classes to perform classification over. 10 by default.
    :param epochs: Number of epochs to train the weights over. 3 by default.
    :param write_results: True to write mean errors and std devs to file. False by default.
    """
    _20_d_stars = np.zeros(num_of_runs)
    _20_test_error_rates = np.zeros(num_of_runs)

    for i in tqdm(range(num_of_runs)):

        # ------- T R A I N: CROSS VALIDATION TO GET D-STAR ----------------------------------------------------------

        # MAKE TRAINING & TEST DATASET SPLITS:
        _80split, _20split = train_test_split(ds, test_size=0.2, random_state=i)

        mean_val_error_per_d = dict()

        # USE 5-FOLD CROSS VALIDATION TO FIND BEST DEGREE `d_star`:---------------------------------------------------
        for degree in degrees:
            mean_val_error_per_d = _calc_mean_error_per_deg_by_5f_cv(mean_val_error_per_d, _80split, epochs, degree,
                                                                     k_classes)
            print(f'For degree {degree} the mean_kfold_val_error = {mean_val_error_per_d[degree]}')

        # DEGREE WITH LOWEST MEAN VALIDATION ERROR:
        d_star = min(mean_val_error_per_d, key=lambda k: mean_val_error_per_d[k])
        _20_d_stars[i] = d_star

        # RE-TRAIN WEIGHTS BUT WITH WHOLE 80% TRAIN SET AND USING `d_star` -----------------------------------------

        # GET DATA POINTS AND LABELS FROM TRAINING SET SPLIT:
        x_train_80 = _80split[:, 1:]
        y_train_80 = _80split[:, 0]

        # PRE-COMPUTE KERNEL MATRIX:
        kernel_matrix_train_80 = _compute_polykern_matrix(x1=x_train_80, x2=x_train_80, degree=d_star)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_80, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_80, reps=(epochs, epochs))

        # TRAIN WEIGHTS (not interested in this error):
        _, trained_alpha = train_kp(y=y, k_mat=k_mat, degree=d_star, k_classes=k_classes)

        # TRAIN WEIGHTS BUT WITH WHOLE 80% TRAIN SET AND USING `d_star` -----------------------------------------

        # GET DATA POINTS AND LABELS FROM TEST SET SPLIT:
        x_test_20 = _20split[:, 1:]
        y_test_20 = _20split[:, 0]

        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_train_80_test_20 = _compute_polykern_matrix(x1=x_train_80, x2=x_test_20, degree=d_star)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_train_80_test_20 = kernel_matrix_train_80_test_20.T
        k_mat = np.tile(A=kernel_matrix_train_80_test_20, reps=epochs)

        # TEST WITH TRAINED WEIGHTS & CALC ERRORS:
        test_error_prct = test_kp(k_mat=k_mat, trained_alpha=trained_alpha, y=y_test_20, degree=d_star)
        _20_test_error_rates[i] = test_error_prct
        print(f'd_star {d_star} has test error {test_error_prct}')

    # CALC MEANS ACROSS 20 RUNS
    mean_d_star = np.mean(_20_d_stars)
    stddev_d_star = np.std(_20_d_stars)
    mean_test_errors = np.mean(_20_test_error_rates)
    stddev_test_erros = np.std(_20_test_error_rates)

    # (OPTIONAL) WRITE MEANS TO FILE:
    if write_results:
        dir = f'../saved_values'
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f'../saved_values/means.txt', 'w') as f:
            f.write(f'mean_d_star={mean_d_star}\n')
            f.write(f'stddev_d_star={stddev_d_star}\n')
            f.write(f'mean_test_errors={mean_test_errors}\n')
            f.write(f'stddev_test_erros={stddev_test_erros}\n')


if __name__ == '__main__':

    start_time = time.time()
    ds = np.loadtxt('../../datasets/zipcombo.dat')
    # ds = ds[:100, :]
    print(f'ds.shape = {ds.shape}')

    run_cv_kp(ds=ds, degrees=range(3, 8), num_of_runs=20, k_classes=10, epochs=3, write_results=True)

    mins = (time.time() - start_time) / 60

    print(f'time taken = {round(mins, 4)} minutes')
