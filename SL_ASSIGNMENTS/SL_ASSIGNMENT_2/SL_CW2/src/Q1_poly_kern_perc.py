import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import os
import shared_functions as shfun

# ---------- QUESTION 1. MEAN & STD DEV OF ERROR RATES  --------------------------------------------------------------


def run_kp(zipcombo, degree, epochs, k_classes=10, num_of_runs=20, write_results=False):
    """
    Train weights (`alpha`) on 80% dataset splits for given number of epochs. Run test with trained weights on
    corresponding 20% dataset split. Compute error rates +/- std dev for each split for both training and test split
    averaged over given number of independent runs.
    :param zipcombo: Full dataset.
    :param degree: Degree for polynomial kernel evaluation. (Expected to be 1,2,3,4,5,6,7).
    :param epochs: Number of epochs to train the weights over. 3 by default.
    :param k_classes: Number of classes to perform classification over. 10 by default.
    :param num_of_runs: Number of independent runs. 20 by default.
    :param write_results: True to write mean errors and std devs to file. False by default.
    """
    error_rates_train = np.zeros(num_of_runs)
    error_rates_test = np.zeros(num_of_runs)

    for i in tqdm(range(num_of_runs)):

        # ---------------------------------------- Q1. T R A I N -----------------------------------------------------

        # MAKE TRAINING & TEST DATASET SPLITS:
        zipcombo_80split, zipcombo_20split = train_test_split(zipcombo, test_size=0.2, random_state=i)

        # GET DATA POINTS AND LABELS FROM TRAINING SET SPLIT:
        x_train_80 = zipcombo_80split[:, 1:]
        y_train_80 = zipcombo_80split[:, 0]

        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_train_80 = shfun.compute_polykern_matrix(x1=x_train_80, x2=x_train_80, degree=degree)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_80, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_80, reps=(epochs, epochs))

        # TRAIN WEIGHTS & CALC ERRORS:
        error_rate_prct_train, trained_alpha = shfun.train_kp(y=y, k_mat=k_mat, degree=degree, k_classes=k_classes)
        error_rates_train[i] = error_rate_prct_train

        # (OPTIONAL) WRITE EACH RESULT OUT TO FILE:
        if write_results:
            dir = f'../saved_values/d{degree}'
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(f'../saved_values/d{degree}/error_rate_prct_train_run#{i + 1}.txt', 'w') as f:
                f.write(str(error_rate_prct_train))
            np.savetxt(f'../saved_values/d{degree}/alpha_vec#{i+1}.csv', trained_alpha)

        # ----------------------------------------- Q1. T E S T ------------------------------------------------------

        # GET DATA POINTS AND LABELS FROM TEST SET SPLIT:
        x_test_20 = zipcombo_20split[:, 1:]
        y_test_20 = zipcombo_20split[:, 0]

        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_train_80_test_20 = shfun.compute_polykern_matrix(x1=x_train_80, x2=x_test_20, degree=degree)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_train_80_test_20 = kernel_matrix_train_80_test_20.T
        k_mat = np.tile(A=kernel_matrix_train_80_test_20, reps=epochs)

        # TEST WITH TRAINED WEIGHTS & CALC ERRORS:
        error_rate_prct_test = shfun.test_kp(ds=ds, k_mat=k_mat, trained_alpha=trained_alpha, y=y_test_20)
        error_rates_test[i] = error_rate_prct_test

        # (OPTIONAL) WRITE EACH RESULT OUT TO FILE:
        if write_results:
            with open(f'../saved_values/d{degree}/error_rate_prct_test_run#{i + 1}.txt', 'w') as f:
                f.write(str(error_rate_prct_train))

    # ---------- Q1. MEAN ERRORS & STD DEVS ---------------------------------------------------------------------------

    # AFTER ALL 20 RUNS, CALC MEANS & STD DEVS:
    mean_train_error = np.mean(error_rates_train)
    stdev_train_error = np.std(error_rates_train)
    mean_test_error = np.mean(error_rates_test)
    stdev_test_error = np.std(error_rates_test)

    # PRINT MEANS:
    print(f'mean train error={mean_train_error}, stdev train error={stdev_train_error}\n')
    print(f'mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')

    # (OPTIONAL) WRITE MEANS TO FILE:
    if write_results:
        with open(f'../saved_values/d{degree}/mean_stddev_train_error.txt', 'w') as f1:
            f1.write(f'degree {degree}, mean train error={mean_train_error}\nstdev train error={stdev_train_error}\n')
        with open(f'../saved_values/d{degree}/mean_stddev_test_error.txt', 'w') as f2:
            f2.write(f'degree {degree}, mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')


if __name__ == '__main__':

    start_time = time.time()
    ds = np.loadtxt('../../datasets/zipcombo.dat')
    ds = ds[:100, :]
    print(f'ds.shape = {ds.shape}')

    for i in range(7):
        degree = i + 1
        print(f'degree: {i + 1}')
        run_kp(zipcombo=ds, degree=degree, num_of_runs=20, k_classes=10, epochs=3, write_results=True)

    print(f'time taken = {round(time.time() - start_time, 4)} seconds')
