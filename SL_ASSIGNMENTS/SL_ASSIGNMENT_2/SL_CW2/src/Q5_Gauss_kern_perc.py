import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import os
import shared_functions as shfun


def test_kp(k_mat, trained_alpha, y, c):
    """
    Perform predictions on test dataset split and calculate numbers of mis-classifications using given kernel matrix
    and pre-trained weights from corresponding training dataset split. (All given values are epoch-tiled).
    :param k_mat: Pre-computed kernel matrix (epoch-tiled) for corresponding training dataset split and test set split.
    :param trained_alpha: Trained weights (epoch-tiled), trained on corresponding training dataset split.
    :param y: Labels (epoch-tiled). 1d array.
    :param c: Width to use for Gaussian kernel evaluation.
    :return: The error rate percentage for given dataset split.
    """
    # USE TRAINED WEIGHTS `trained_alpha` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    k_mat = k_mat.T  # non-symmetric kernel matrix needs correct orientation for dot product with trained_alpha
    mistakes = shfun.predict_with_trained_alpha(trained_alpha, k_mat, y)
    print(f'Number of test mistakes for c {c} = {mistakes}')
    error_rate_prct = mistakes / len(ds)
    error_rate_prct *= 100
    print(f'Test error for c {c} = {error_rate_prct} %')
    return error_rate_prct


def train_kp(y, k_mat, c, k_classes: int):
    """
    Perform a form of online learning by updating the weight (`alpha`) by updating its value the weighted kernel
    evaluation whenever its prediction is wrong.
    :param y: Labels for training dataset split of MNIST. As 1d array with shape (m, 257) where `m` is number of
    datapoints (epoch-tiled).
    :param k_mat: Kernel matrix for given training dataset (epoch-tiled).
    :param c: Width to use for Gaussian kernel evaluation.
    :param k_classes: Number of classes for which to perform classification.
    :return: The error rate percentage and the trained weights for the given dataset, width, k_classes and epochs.
    """
    y_vec = shfun.convert_y_to_vector(y, k_classes)
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

    # AFTER ALL EPOCHS, USE TRAINED WEIGHTS `alpha_vec` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    mistakes = shfun.predict_with_trained_alpha(alpha_vec, k_mat, y)
    print(f'Number of training mistakes {mistakes}')

    error_rate_prct = 100 * (mistakes / m)
    print(f'Train error for c {c} = {error_rate_prct} %')
    return error_rate_prct, alpha_vec


def _calc_mean_error_per_c_by_5f_cv(mean_val_error_per_c, _80split, epochs, width, k_classes):
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
        kernel_matrix_train_cv = shfun.compute_gauss_kern_matrix(x1=x_train_cv, x2=x_train_cv, c=width)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_cv, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_cv, reps=(epochs, epochs))

        # TRAIN WEIGHTS (not interested in error rate from training fold):
        _, trained_alpha = train_kp(y=y, k_mat=k_mat, c=width, k_classes=k_classes)

        # USE TRAINED WEIGHTS TO RUN *VALIDATION* TESTS. RECORD ERROR FOR THIS WIDTH c:
        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_val = shfun.compute_gauss_kern_matrix(x1=x_train_cv, x2=x_val_cv, c=width)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_val = kernel_matrix_val.T
        k_mat = np.tile(A=kernel_matrix_val, reps=epochs)

        # VALIDATION TEST WITH TRAINED WEIGHTS & CALC ERRORS FOR THIS WIDTH c:
        error_prct_kfolds[i] = test_kp(k_mat=k_mat, trained_alpha=trained_alpha, y=y_val_cv, c=width)
    # AFTER ALL 5 FOLDS, CALC MEANS OF VALIDATION TESTS FOR THIS WIDTH c:
    # RECORD MEAN VALIDATION ERROR PER WIDTH c
    mean_val_error_per_c[width] = np.mean(error_prct_kfolds)

    return mean_val_error_per_c


def run_gauss_cv_kp(ds, s, num_of_runs=20, k_classes=10, epochs=3, write_results=False):
    """
    Train weights (`alpha`) on 80% dataset splits for given number of epochs. Run test with trained weights on
    corresponding 20% dataset split. Compute error rates +/- std dev for each split for both training and test split
    averaged over given number of independent runs.
    :param ds: Dataset.
    :param s: Range of widths, `c`, to try for computing the Gaussian kernel evaluation.
    :param num_of_runs: Number of independent runs. 20 by default.
    :param k_classes: Number of classes to perform classification over. 10 by default.
    :param epochs: Number of epochs to train the weights over. 3 by default.
    :param write_results: True to write mean errors and std devs to file. False by default.
    """
    _20_c = np.zeros(num_of_runs)
    _20_test_error_rates = np.zeros(num_of_runs)

    for i in tqdm(range(num_of_runs)):

        # ------- T R A I N: CROSS VALIDATION TO FIND BEST C ----------------------------------------------------------

        # MAKE TRAINING & TEST DATASET SPLITS:
        _80split, _20split = train_test_split(ds, test_size=0.2, random_state=i)

        mean_val_error_per_c = dict()

        # USE 5-FOLD CROSS VALIDATION TO FIND BEST WIDTH: ------------------------------------------------------------
        for c in s:
            mean_val_error_per_c = _calc_mean_error_per_c_by_5f_cv(mean_val_error_per_c, _80split, epochs, c, k_classes)
            print(f'For width {c}, the mean_kfold_val_error = {mean_val_error_per_c[c]}')

        # WIDTH WITH LOWEST MEAN VALIDATION ERROR:
        c_star = min(mean_val_error_per_c, key=lambda k: mean_val_error_per_c[k])
        _20_c[i] = c_star

        # RE-TRAIN WEIGHTS BUT WITH WHOLE 80% TRAIN SET AND USING `c_star` -----------------------------------------

        # GET DATA POINTS AND LABELS FROM TRAINING SET SPLIT:
        x_train_80 = _80split[:, 1:]
        y_train_80 = _80split[:, 0]

        # PRE-COMPUTE KERNEL MATRIX:
        kernel_matrix_train_80 = shfun.compute_gauss_kern_matrix(x1=x_train_80, x2=x_train_80, c=c_star)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_80, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_80, reps=(epochs, epochs))

        # TRAIN WEIGHTS (not interested in this error):
        _, trained_alpha = train_kp(y=y, k_mat=k_mat, c=c_star, k_classes=k_classes)

        # TEST ON 20% DATASET WITH TRAINED WEIGHTS FROM 80% DATASET AND USING `c_star` ------------------------------

        # GET DATA POINTS AND LABELS FROM TEST SET SPLIT:
        x_test_20 = _20split[:, 1:]
        y_test_20 = _20split[:, 0]

        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_train_80_test_20 = shfun.compute_gauss_kern_matrix(x1=x_train_80, x2=x_test_20, c=c_star)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_train_80_test_20 = kernel_matrix_train_80_test_20.T
        k_mat = np.tile(A=kernel_matrix_train_80_test_20, reps=epochs)

        # TEST WITH TRAINED WEIGHTS & CALC ERRORS:
        test_error_prct = test_kp(k_mat=k_mat, trained_alpha=trained_alpha, y=y_test_20, c=c_star)
        _20_test_error_rates[i] = test_error_prct
        print(f'c_star {c_star} has test error {test_error_prct}')

    # CALC MEANS ACROSS 20 RUNS
    mean_c_star = np.mean(_20_c)
    stddev_c_star = np.std(_20_c)
    mean_test_errors = np.mean(_20_test_error_rates)
    stddev_test_errors = np.std(_20_test_error_rates)

    # (OPTIONAL) WRITE MEANS TO FILE:
    if write_results:
        dir = f'../saved_values'
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir = f'../saved_values/Q5_2'
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(f'../saved_values/Q5_2/means.txt', 'w') as f:
            f.write(f'mean_c_star={mean_c_star}\n')
            f.write(f'stddev_c_star={stddev_c_star}\n')
            f.write(f'mean_test_errors={mean_test_errors}\n')
            f.write(f'stddev_test_errors={stddev_test_errors}\n')


def run_gauss_kp(ds, c, epochs, k_classes=10, num_of_runs=20, write_results=True):
    """
    Train weights (`alpha`) on 80% dataset splits for given number of epochs. Run test with trained weights on
    corresponding 20% dataset split. Compute error rates +/- std dev for each split for both training and test split
    averaged over given number of independent runs.
    :param ds: Full dataset.
    :param c: Width, `c`, to try for computing the Gaussian kernel evaluation.
    :param epochs: Number of epochs to train the weights over. 3 by default.
    :param k_classes: Number of classes to perform classification over. 10 by default.
    :param num_of_runs: Number of independent runs. 20 by default.
    :param write_results: True to write 20 train and test error rates and std devs to file. True by default.
    """
    _20_error_rates_train = np.zeros(num_of_runs)
    _20_error_rates_test = np.zeros(num_of_runs)

    for i in tqdm(range(num_of_runs)):

        # ---------------------------------------- Q 5.1. T R A I N --------------------------------------------------

        # MAKE TRAINING & TEST DATASET SPLITS:
        ds_80split, ds_20split = train_test_split(ds, test_size=0.2, random_state=i)

        # GET DATA POINTS AND LABELS FROM TRAINING SET SPLIT:
        x_train_80 = ds_80split[:, 1:]
        y_train_80 = ds_80split[:, 0]

        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_train_80 = shfun.compute_gauss_kern_matrix(x1=x_train_80, x2=x_train_80, c=c)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_80, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_80, reps=(epochs, epochs))

        # TRAIN WEIGHTS & CALC ERRORS:
        error_rate_prct_train, trained_alpha = train_kp(y=y, k_mat=k_mat, c=c, k_classes=k_classes)
        _20_error_rates_train[i] = error_rate_prct_train

        # ----------------------------------------- Q 5.1. T E S T --------------------------------------------------

        # GET DATA POINTS AND LABELS FROM TEST SET SPLIT:
        x_test_20 = ds_20split[:, 1:]
        y_test_20 = ds_20split[:, 0]

        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_train_80_test_20 = shfun.compute_gauss_kern_matrix(x1=x_train_80, x2=x_test_20, c=c)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_train_80_test_20 = kernel_matrix_train_80_test_20.T
        k_mat = np.tile(A=kernel_matrix_train_80_test_20, reps=epochs)

        # TEST WITH TRAINED WEIGHTS & CALC ERRORS:
        error_rate_prct_test = test_kp(k_mat=k_mat, trained_alpha=trained_alpha, y=y_test_20, c=c)
        _20_error_rates_test[i] = error_rate_prct_test

    # ---------- Q1. MEAN ERRORS & STD DEVS ---------------------------------------------------------------------------

    # AFTER ALL 20 RUNS, CALC MEANS & STD DEVS:
    mean_train_error = np.mean(_20_error_rates_train)
    stdev_train_error = np.std(_20_error_rates_train)
    mean_test_error = np.mean(_20_error_rates_test)
    stdev_test_error = np.std(_20_error_rates_test)

    # PRINT MEANS:
    print(f'mean train error={mean_train_error}, stdev train error={stdev_train_error}\n')
    print(f'mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')

    # ROUND TO 2 DECIMAL PLACES:
    mean_train_error = round(mean_train_error, 2)
    stdev_train_error = round(stdev_train_error, 2)
    mean_test_error = round(mean_test_error, 2)
    stdev_test_error = round(stdev_test_error, 2)

    # (OPTIONAL) WRITE MEANS TO FILE:
    if write_results:
        dir = f'../saved_values'
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir = f'../saved_values/Q5_1'
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir = f'../saved_values/Q5_1/c_{c}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f'../saved_values/Q5_1/c_{c}/mean_stddev_train_error.txt', 'w') as f1:
            f1.write(f'c {c}, mean train error={mean_train_error}\nstdev train error={stdev_train_error}\n')
        with open(f'../saved_values/Q5_1/c_{c}/mean_stddev_test_error.txt', 'w') as f2:
            f2.write(f'c {c}, mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')


if __name__ == '__main__':

    start_time = time.time()
    ds = np.loadtxt('../../datasets/zipcombo.dat')
    # ds = ds[:100, :]
    print(f'ds.shape = {ds.shape}')
    # s = [0.0001, 0.001, 0.01, 0.1, 1, 10, 50]
    # for c in s:
    #     print(f'c: {c}')
    #     run_gauss_kp(ds=ds, c=c, num_of_runs=20, k_classes=10, epochs=3, write_results=True)

    s = [0.001, 0.01, 0.1, 1]
    run_gauss_cv_kp(ds, s=s, num_of_runs=20, k_classes=10, epochs=3, write_results=True)

    print(f'time taken = {round((time.time() - start_time)/60, 4)} minutes')
