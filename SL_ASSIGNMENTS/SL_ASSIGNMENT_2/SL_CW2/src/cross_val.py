import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import os
import shared_functions as shfun


# ---------- QUESTION 2. Cross validation to find best d ------------------------------------------------------------


def run_cv_kp(ds, degrees, num_of_runs=20, k_classes=10, epochs=3, write_results=False):
    """
    Train weights (`alpha`) on 80% dataset splits for given number of epochs. Find the best degree (`d_star`) for the
    polynomial kernel evaluation by 5-fold cross validation of this split.
    With best degree (`d_star`), run test with trained weights on corresponding 20% dataset split. Compute error
    rates +/- std dev for each split for both training and test split averaged over given number of independent runs.
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
            mean_val_error_per_d = shfun.calc_mean_error_per_deg_by_5f_cv(ds, mean_val_error_per_d, _80split, epochs,
                                                                          degree, k_classes)
            print(f'For degree {degree} the mean_kfold_val_error = {mean_val_error_per_d[degree]}')

        # DEGREE WITH LOWEST MEAN VALIDATION ERROR:
        d_star = min(mean_val_error_per_d, key=lambda k: mean_val_error_per_d[k])
        _20_d_stars[i] = d_star

        # RE-TRAIN WEIGHTS BUT WITH WHOLE 80% TRAIN SET AND USING `d_star` -----------------------------------------

        # GET DATA POINTS AND LABELS FROM TRAINING SET SPLIT:
        x_train_80 = _80split[:, 1:]
        y_train_80 = _80split[:, 0]

        # PRE-COMPUTE KERNEL MATRIX:
        kernel_matrix_train_80 = shfun.compute_polykern_matrix(x1=x_train_80, x2=x_train_80, degree=d_star)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_80, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_80, reps=(epochs, epochs))

        # TRAIN WEIGHTS (not interested in this error):
        _, trained_alpha = shfun.train_kp(y=y, k_mat=k_mat, degree=d_star, k_classes=k_classes)

        # TEST ON 20% DATASET WITH TRAINED WEIGHTS FROM 80% DATASET AND USING `d_star` ------------------------------

        # GET DATA POINTS AND LABELS FROM TEST SET SPLIT:
        x_test_20 = _20split[:, 1:]
        y_test_20 = _20split[:, 0]

        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_train_80_test_20 = shfun.compute_polykern_matrix(x1=x_train_80, x2=x_test_20, degree=d_star)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_train_80_test_20 = kernel_matrix_train_80_test_20.T
        k_mat = np.tile(A=kernel_matrix_train_80_test_20, reps=epochs)

        # TEST WITH TRAINED WEIGHTS & CALC ERRORS:
        test_error_prct = shfun.test_kp(ds=ds, k_mat=k_mat, trained_alpha=trained_alpha, y=y_test_20, degree=d_star)
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
