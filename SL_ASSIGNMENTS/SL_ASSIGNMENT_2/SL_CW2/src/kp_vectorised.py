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


def _precompute_polykern_matrix(x1, x2, degree):
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


# ---------- QUESTION 1. MEAN & STD DEV OF ERROR RATES  --------------------------------------------------------------

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
    k_mat = k_mat.T  # non-symmetric kernel matrix needs right orientation for dot product with trained_alpha
    mistakes = _predict_with_trained_alpha(trained_alpha, k_mat, y)
    print(f'Number of test mistakes for degree {degree} = {mistakes}')

    error_rate_prct = 100 * (mistakes / len(ds))
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

    # AFTER ALL EPOCHS, USE TRAINED WEIGHTS `alpha_vec` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    mistakes = _predict_with_trained_alpha(alpha_vec, k_mat, y)
    print(f'Number of training mistakes {mistakes}')

    error_rate_prct = 100 * (mistakes / m)
    print(f'Train error for degree {degree} = {error_rate_prct} %')
    return error_rate_prct, alpha_vec


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
        kernel_matrix_train_80 = _precompute_polykern_matrix(x1=x_train_80, x2=x_train_80, degree=degree)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_80, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_80, reps=(epochs, epochs))

        # TRAIN WEIGHTS & CALC ERRORS:
        error_rate_prct_train, trained_alpha = train_kp(y=y, k_mat=k_mat, degree=degree, k_classes=k_classes)
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
        kernel_matrix_train_80_test_20 = _precompute_polykern_matrix(x1=x_train_80, x2=x_test_20, degree=degree)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_train_80_test_20 = kernel_matrix_train_80_test_20.T
        k_mat = np.tile(A=kernel_matrix_train_80_test_20, reps=epochs)

        # TEST WITH TRAINED WEIGHTS & CALC ERRORS:
        error_rate_prct_test = test_kp(k_mat=k_mat, trained_alpha=trained_alpha, y=y_test_20)
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
