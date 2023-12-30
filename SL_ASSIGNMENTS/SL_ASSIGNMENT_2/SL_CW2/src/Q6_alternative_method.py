import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import shared_functions as shfun
from tqdm import tqdm
from time import time
import os


# ---------- QUESTION 6. ALTERNATIVE METHOD: DETERMINE CONFUSION RATES & MIS-CLASSIFICATIONS OF GAUSSIAN KERNEL -------

def _visualise_confusion_matrices():
    """
    Build and display confusion matrix for mean values and standard deviations for digits confused for other digits
    according to their true labels.
    """
    mean_confs = np.loadtxt('../saved_values/Q6/mean_confs.txt')
    # VALUES NEED TO BE SCALED UP FOR VISUALISATION REASONS (OTHERWISE YOU JUST SEE 0.00 IN MOST CELLS)
    mean_confs = np.round(mean_confs * 100, 2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_confs, annot=True, fmt=".2f", cmap='Greys')
    plt.title('mean confusions')
    plt.ylabel('true y')
    plt.xlabel('predicted y')
    plt.show()

    stddev_confs = np.loadtxt('../saved_values/Q6/stddev_confs.txt')
    # VALUES NEED TO BE SCALED UP FOR VISUALISATION REASONS (OTHERWISE YOU JUST SEE 0.00 IN MOST CELLS)
    stddev_confs = np.round(stddev_confs * 100, 2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(stddev_confs, annot=True, fmt=".2f", cmap='Greys')
    plt.title('std dev of confusions')
    plt.ylabel('true y')
    plt.xlabel('predicted y')
    plt.show()



def _predict_with_trained_alpha_confusions(trained_alpha, k_mat, y):
    """

    :param trained_alpha: Trained weights. 2d array.
    :param k_mat: Kernel matrix (epoch-tiled). 2d array.
    :param y: Labels (epoch-tiled). 1d array.
    :return: confusion matrix and rate
    """
    # FOR EACH EPOCH, USE TRAINED WEIGHTS `alpha_vec` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    predictions_with_trained_alpha = np.dot(trained_alpha, k_mat)
    # digit & index happen to be same number for 10 classes with 0-based indexing.
    predicted_y = np.argmax(predictions_with_trained_alpha, axis=0).reshape(-1, 1)
    true_y = y.reshape(-1, 1).astype(np.int32)
    # one_hot_indices_of_confs = -1 * np.ones(len(y))
    indices_of_confusions = np.where(predicted_y != true_y)[0]
    # one_hot_indices_of_confs[indices_of_confusions] = 1
    mismatch_indices = np.where(true_y != predicted_y)[0]
    confusion_matrix = np.zeros((10, 10), dtype=int)
    np.add.at(confusion_matrix, (true_y[mismatch_indices], predicted_y[mismatch_indices]), 1)
    occurrences = np.bincount(true_y.flatten(), minlength=10).reshape(-1, 1).astype(np.float32)
    occurrences += np.finfo(np.float64).eps  # avoid divide by zero
    confusion_matrix_rate = confusion_matrix / occurrences
    true_and_pred = np.concatenate([true_y, predicted_y], axis=1)  # to let me view more easily side-by-side in IDE
    # return confusion_matrix, confusion_matrix_rate, one_hot_indices_of_confs
    return confusion_matrix, confusion_matrix_rate


def _test_kp_confusions(k_mat, trained_alpha, y):
    """

    :param k_mat: Pre-computed kernel matrix (epoch-tiled) for corresponding training dataset split and test set split.
    :param trained_alpha: Trained weights (epoch-tiled), trained on corresponding training dataset split.
    :param y: Labels (epoch-tiled). 1d array.
    :return: The error rate percentage for given dataset split.
    """
    # USE TRAINED WEIGHTS `trained_alpha` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    k_mat = k_mat.T  # non-symmetric kernel matrix needs correct orientation for dot product with trained_alpha
    # confusion_matrix, confusion_matrix_rate, one_hot_indices_of_confs = \
    #     _predict_with_trained_alpha_confusions(trained_alpha, k_mat, y)
    confusion_matrix, confusion_matrix_rate, = _predict_with_trained_alpha_confusions(trained_alpha, k_mat, y)
    # return confusion_matrix, confusion_matrix_rate, one_hot_indices_of_confs
    return confusion_matrix, confusion_matrix_rate


# ---------- QUESTION 3. Cross validation to find best d ------------------------------------------------------------


def run_cv_kp_confs(ds, c=0.01, num_of_runs=20, k_classes=10, epochs=3, write_results=True):
    """

    :param ds: Dataset.
    :param c:
    :param num_of_runs: Number of independent runs. 20 by default.
    :param k_classes: Number of classes to perform classification over. 10 by default.
    :param epochs: Number of epochs to train the weights over. 3 by default.
    :param write_results: True to write mean errors and std devs to file. False by default.
    """
    _20_confusion_matrix_rates = np.zeros((k_classes, k_classes, num_of_runs))
    _20_d_stars = np.zeros(num_of_runs)

    for i in tqdm(range(num_of_runs)):

        # MAKE TRAINING & TEST DATASET SPLITS:
        _80split, _20split = train_test_split(ds, test_size=0.2, random_state=i)

        # GET DATA POINTS AND LABELS FROM TRAINING SET SPLIT:
        x_train_80 = _80split[:, 1:]
        y_train_80 = _80split[:, 0]

        # PRE-COMPUTE KERNEL MATRIX:
        kernel_matrix_train_80 = shfun.compute_gauss_kern_matrix(x1=x_train_80, x2=x_train_80, c=c)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        y = np.tile(A=y_train_80, reps=epochs)
        k_mat = np.tile(A=kernel_matrix_train_80, reps=(epochs, epochs))

        # TRAIN WEIGHTS (not interested in this error):
        _, trained_alpha = shfun.train_kp(y=y, k_mat=k_mat, d_or_c=c, k_classes=k_classes)

        # TEST ON 20% DATASET WITH TRAINED WEIGHTS FROM 80% DATASET AND USING `d_star` ------------------------------

        # GET DATA POINTS AND LABELS FROM TEST SET SPLIT:
        x_test_20 = _20split[:, 1:]
        y_test_20 = _20split[:, 0]

        # PRECOMPUTE KERNEL MATRIX:
        kernel_matrix_train_80_test_20 = shfun.compute_gauss_kern_matrix(x1=x_train_80, x2=x_test_20, c=c)

        # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
        kernel_matrix_train_80_test_20 = kernel_matrix_train_80_test_20.T
        k_mat = np.tile(A=kernel_matrix_train_80_test_20, reps=epochs)

        # TEST WITH TRAINED WEIGHTS & CALC ERRORS:
        # confusion_matrix, confusion_matrix_rate, one_hot_indices_of_confs = \
        #     test_kp_confusions(k_mat=k_mat, trained_alpha=trained_alpha, y=y_test_20)
        confusion_matrix, confusion_matrix_rate, = _test_kp_confusions(k_mat=k_mat, trained_alpha=trained_alpha,
                                                                       y=y_test_20)

        _20_confusion_matrix_rates[:, :, i] = confusion_matrix_rate

    # CALC MEANS ACROSS 20 RUNS
    mean_confusions = np.mean(_20_confusion_matrix_rates, axis=2)
    stddev_confusions = np.std(_20_confusion_matrix_rates, axis=2)

    total_conf_rates_per_digit = np.sum(mean_confusions, axis=1)

    # (OPTIONAL) WRITE MEANS TO FILE:
    if write_results:
        dir = f'../saved_values'
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir = f'../saved_values/Q6'
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.savetxt('../saved_values/Q6/mean_confs.txt', mean_confusions, fmt='%.4f')
        np.savetxt('../saved_values/Q6/stddev_confs.txt', stddev_confusions, fmt='%.4f')
        np.savetxt('../saved_values/Q6/total_conf_rates_per_digit.txt', total_conf_rates_per_digit, fmt='%.4f')


if __name__ == '__main__':

    # _visualise_confusion_matrices()

    # mean_confs = np.loadtxt('../saved_values/mean_confs.txt')
    # mean_confs_sums = np.sum(mean_confs, axis=1)
    # pass

    start_time = time()
    ds = np.loadtxt('../../datasets/zipcombo.dat')
    # ds = ds[:100, :]
    print(f'ds.shape = {ds.shape}')

    # run_cv_kp_confs(ds=ds)

    _visualise_confusion_matrices()
    mins = (time() - start_time) / 60

    print(f'time taken = {round(mins, 4)} minutes')

