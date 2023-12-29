import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import shared_functions as shfun

# ---------- QUESTION 3. FIND MEAN CONFUSIONS AND DISPLAY CONFUSION MATRIX -------------------------------------------


def _visualise_confusion_matrices():
    """
    Build and display confusion matrix for mean values and standard deviations for digits confused for other digits
    according to their true labels.
    """
    mean_confs = np.loadtxt('../saved_values/mean_confs.txt')
    stddev_confs = np.loadtxt('../saved_values/stddev_confs.txt')
    plt.figure(figsize=(10, 8))  # Size it as needed
    sns.heatmap(mean_confs, annot=True, fmt=".2f", cmap='Greys')  # or cmap='coolwarm', 'Blues', etc.
    plt.title('mean confusions')
    plt.ylabel('true y')
    plt.xlabel('predicted y')
    plt.show()

    sns.heatmap(stddev_confs, annot=True, fmt=".2f", cmap='Greys')  # or cmap='coolwarm', 'Blues', etc.
    plt.title('std dev of confusions')
    plt.ylabel('true y')
    plt.xlabel('predicted y')
    plt.show()


def _predict_with_trained_alpha_confusions(trained_alpha, k_mat, y):
    """
    Compute predictions with trained weights and kernel matrix.
    Record confusions, such that (7,2) indicates that 7 was the true label, 2 was the predicted label.
    :param trained_alpha: Trained weights. 2d array.
    :param k_mat: Kernel matrix (epoch-tiled). 2d array.
    :param y: Labels (epoch-tiled). 1d array.
    :return: Number of mistakes made in prediction.
    """
    # FOR EACH EPOCH, USE TRAINED WEIGHTS `alpha_vec` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    predictions_with_trained_alpha = np.dot(trained_alpha, k_mat)
    # digit & index happen to be same number for 10 classes with 0-based indexing.
    predicted_y = np.argmax(predictions_with_trained_alpha, axis=0).reshape(-1, 1)
    true_y = y.reshape(-1, 1).astype(np.int32)
    one_hot_indices_of_confs = -1 * np.ones(len(y))
    indices_of_confusions = np.where(predicted_y != true_y)[0]
    one_hot_indices_of_confs[indices_of_confusions] = 1
    mismatch_indices = np.where(true_y != predicted_y)[0]
    confusion_matrix = np.zeros((10, 10), dtype=int)
    np.add.at(confusion_matrix, (true_y[mismatch_indices], predicted_y[mismatch_indices]), 1)
    occurrences = np.bincount(true_y.flatten(), minlength=10).reshape(-1, 1).astype(np.float32)
    occurrences += np.finfo(np.float64).eps  # avoid divide by zero
    confusion_matrix_rate = confusion_matrix / occurrences
    true_and_pred = np.concatenate([true_y, predicted_y], axis=1)  # to let me view more easily side-by-side in IDE
    return confusion_matrix, confusion_matrix_rate, one_hot_indices_of_confs


def test_kp_confusions(k_mat, trained_alpha, y, degree):
    """
    Perform predictions on test dataset split and calculate numbers of misclassifications using given kernel matrix
    and pre-trained weights from corresponding training dataset split. (All given values are epoch-tiled).
    :param k_mat: Pre-computed kernel matrix (epoch-tiled) for corresponding training dataset split and test set split.
    :param trained_alpha: Trained weights (epoch-tiled), trained on corresponding training dataset split.
    :param y: Labels (epoch-tiled). 1d array.
    :param degree: Degree to use for polynomial kernel evaluation.
    :return: The error rate percentage for given dataset split.
    """
    # USE TRAINED WEIGHTS `trained_alpha` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    k_mat = k_mat.T  # non-symmetric kernel matrix needs correct orientation for dot product with trained_alpha
    confusion_matrix, confusion_matrix_rate, one_hot_indices_of_confs = _predict_with_trained_alpha_confusions(trained_alpha, k_mat, y)
    return confusion_matrix, confusion_matrix_rate, one_hot_indices_of_confs


# ---------- QUESTION 3. Cross validation to find best d ------------------------------------------------------------


def run_cv_kp_confs(ds, degrees, num_of_runs=20, k_classes=10, epochs=3, write_results=False):
    """
    Train weights (`alpha`) on 80% dataset splits for given number of epochs. Find the best degree (`d_star`) for the
    polynomial kernel evaluation by 5-fold cross validation of this split.
    With best degree (`d_star`), run test with trained weights on corresponding 20% dataset split.
    Find misclassification pairs ('confusions') for each split over 20 runs, and calculate the mean and standard
    deviations for each confusion across the 20 runs.
    :param ds: Dataset.
    :param degrees: Degrees to try with polynomial kernel evaluation. (Expected to be within range 1 and 7)
    :param num_of_runs: Number of independent runs. 20 by default.
    :param k_classes: Number of classes to perform classification over. 10 by default.
    :param epochs: Number of epochs to train the weights over. 3 by default.
    :param write_results: True to write mean errors and std devs to file. False by default.
    """
    _20_confusion_matrix_rates = np.zeros((k_classes, k_classes, num_of_runs))
    _20_d_stars = np.zeros(num_of_runs)

    for i in tqdm(range(num_of_runs)):

        # ------- T R A I N: CROSS VALIDATION TO GET D-STAR ---------------------------------------------------------

        # MAKE TRAINING & TEST DATASET SPLITS:
        _80split, _20split = train_test_split(ds, test_size=0.2, random_state=i)

        mean_val_error_per_d = dict()

        # USE 5-FOLD CROSS VALIDATION TO FIND BEST DEGREE `d_star`:--------------------------------------------------
        for degree in degrees:
            mean_val_error_per_d = shfun.calc_mean_error_per_deg_by_5f_cv(mean_val_error_per_d, _80split, epochs, degree,
                                                                     k_classes)
            print(f'For degree {degree} the mean_kfold_val_error = {mean_val_error_per_d[degree]}')

        # DEGREE WITH LOWEST MEAN VALIDATION ERROR:
        d_star = min(mean_val_error_per_d, key=lambda k: mean_val_error_per_d[k])
        _20_d_stars[i] = d_star

        # RE-TRAIN WEIGHTS BUT WITH WHOLE 80% TRAIN SET AND USING `d_star` ------------------------------------------

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
        confusion_matrix, confusion_matrix_rate, one_hot_indices_of_confs = \
            test_kp_confusions(k_mat=k_mat, trained_alpha=trained_alpha, y=y_test_20, degree=d_star)

        _20_confusion_matrix_rates[:, :, i] = confusion_matrix_rate

        dir = f'../saved_values'
        if not os.path.exists(dir):
            os.makedirs(dir)
            np.savetxt(f'../saved_values/indices_of_confs_{i}.txt', one_hot_indices_of_confs, fmt='%d')

    # CALC MEANS ACROSS 20 RUNS
    mean_confusions = np.mean(_20_confusion_matrix_rates, axis=2)
    stddev_confusions = np.std(_20_confusion_matrix_rates, axis=2)

    # (OPTIONAL) WRITE MEANS TO FILE:
    if write_results:
        dir = f'../saved_values'
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.savetxt('../saved_values/mean_confs.txt', mean_confusions, fmt='%.4f')
        np.savetxt('../saved_values/stddev_confs.txt', stddev_confusions, fmt='%.4f')


if __name__ == '__main__':

    # _visualise_confusion_matrices()

    # mean_confs = np.loadtxt('../saved_values/mean_confs.txt')
    # mean_confs_sums = np.sum(mean_confs, axis=1)
    # pass

    start_time = time.time()
    ds = np.loadtxt('../../datasets/zipcombo.dat')
    # ds = ds[:100, :]
    print(f'ds.shape = {ds.shape}')

    # run_cv_kp_confs(ds=ds, degrees=range(3, 8), num_of_runs=20, k_classes=10, epochs=3, write_results=True)
    run_cv_kp_confs(ds=ds, degrees=range(4, 6), num_of_runs=20, k_classes=10, epochs=3, write_results=True)

    mins = (time.time() - start_time) / 60

    print(f'time taken = {round(mins, 4)} minutes')
