import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import pandas as pd


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


def _predict_with_train_alpha(alpha_vec, k_mat, y):
    """

    :param alpha_vec:
    :param k_mat:
    :param y:
    :return:
    """
    # FOR EACH EPOCH, USE TRAINED WEIGHTS `alpha_vec` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    predictions_with_trained_alpha = np.dot(alpha_vec, k_mat)
    # digit & index happen to be same number for 10 classes with 0-based indexing.
    predicted_y = np.argmax(predictions_with_trained_alpha, axis=0).reshape(1, -1)
    predicted_y = predicted_y.reshape(-1, 1)
    true_y = y.reshape(-1, 1).astype(np.int32)
    mistakes = predicted_y != true_y
    mistakes = np.sum(mistakes)
    return mistakes


def test_kp(ds, k_mat, trained_alpha):
    """
    Perform predictions and calculate numbers of misclassifications on given test dataset split using pre-trained
    weights from corresponding training dataset split.
    :param ds: The test dataset split of MNIST. As 2d array with shape (m, 257) where `m` is number of datapoints,
    which are hand-drawn digits.
    :param k_mat: Pre-computed kernel matrix for corresponding training dataset split and test set split.
    :param trained_alpha: Trained weights, trained on corresponding training dataset split.
    :return: The error rate percentage for given dataset split.
    """
    m = len(ds)
    y = ds[:, 0]
    assert k_mat.shape[0] == trained_alpha.shape[1]
    assert k_mat.shape[1] == m

    # USE TRAINED WEIGHTS `trained_alpha` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    mistakes = _predict_with_train_alpha(trained_alpha, k_mat, y)
    print(f'Number of test mistakes {mistakes}')

    error_rate_prct = 100 * (mistakes / len(ds))
    print(f'Test error for degree {degree} = {error_rate_prct} %')
    return error_rate_prct


def train_kp(ds, k_mat, degree: int, k_classes: int, epochs: int):
    """
    Perform a form of online learning by updating the weight (`alpha`) by updating its value the weighted kernel
    evaluation whenever its prediction is wrong.
    :param ds: Training dataset split of MNIST. As 2d array with shape (m, 257) where `m` is number of datapoints,
    which are hand-drawn digits.
    :param k_mat: pre-computed kernel matrix for given training dataset.
    :param degree: Degree to use for polynomial kernel evaluation.
    :param k_classes: Number of classes for which to perform classification.
    :param epochs: Number of epochs to run the online learning algorithm over using the given dataset.
    :return: The error rate percentage and the trained weights for the given dataset, degree, k_classes and epochs.
    """
    m = len(ds)
    alpha_vec = np.zeros((k_classes, len(ds)))
    y = ds[:, 0]
    y_vec = _convert_y_to_vector(y, k_classes)
    assert alpha_vec.shape[0] == y_vec.shape[0] == k_classes
    assert k_mat.shape[0] == k_mat.shape[1] == alpha_vec.shape[1] == m

    for epoch in range(epochs):

        preds = np.zeros((k_classes, m))

        # ITERATE THROUGH ALL DATA POINTS:
        for t in range(m):
            a_slice = alpha_vec[:, :t]
            k_slice = k_mat[t, :t].reshape(-1, 1)
            # MAKE PREDICTIONS FOR THIS DATA POINT ACROSS ALL CLASSES:
            pred = np.dot(a_slice, k_slice)

            # UPDATE WEIGHTS FOR ANY THAT MISCLASSIFIED THIS DATA POINT (IF PRODUCT OF TRUE Y AND PREDICTED IS <= 0)
            # STORE WHICH DIGITS ARE MISCLASSIFIED IN BOOLEAN MASK:
            preds[:, t] = pred.reshape(-1)
            prod_y_and_preds_at_t = y_vec[:, t] * preds[:, t]
            misclass_mask_at_t = prod_y_and_preds_at_t <= 0
            learning_rate = 1
            y_vec_with_which_to_update_alpha = learning_rate * y_vec[misclass_mask_at_t, t].reshape(-1)
            alpha_vec[misclass_mask_at_t, t] = y_vec_with_which_to_update_alpha

        mistakes = _predict_with_train_alpha(alpha_vec, k_mat, y)
        print(f'mistakes during training, epoch#{epoch} = {mistakes}')

    # AFTER ALL EPOCHS, USE TRAINED WEIGHTS `alpha_vec` TO MAKE PREDICTIONS AND COUNT MISTAKES:
    mistakes = _predict_with_train_alpha(alpha_vec, k_mat, y)
    print(f'Number of training mistakes {mistakes}')

    error_rate_prct = 100 * (mistakes / len(ds))
    print(f'Train error for degree {degree} = {error_rate_prct} %')
    return error_rate_prct, alpha_vec


def run_kp(zipcombo, degree, epochs, k_classes=10, num_of_runs=20, write_results=True):
    """
    Train weights (`alpha`) on 80% dataset splits for given number of epochs. Run test with trained weights on
    corresponding 20% dataset split. Compute error rates +/- std dev for each split for both training and test split
    averaged over given number of independent runs.
    :param zipcombo: Full dataset.
    :param degree: Degree for polynomial kernel evaluation.
    :param epochs: Number of epochs to train the weights over. 20 by default.
    :param k_classes: Number of classes to perform classification over. 10 by default.
    :param num_of_runs: Number of independent runs of the 7-fold validation.
    :param write_results: True to write mean errors and std devs to file.
    """
    error_rates_train = np.zeros(num_of_runs)
    error_rates_test = np.zeros(num_of_runs)

    for i in tqdm(range(num_of_runs)):
        # serves dual purpose: loop n times, as well as providing seed for unique splits.

        # MAKE TRAINING & TEST DATASET SPLITS
        zipcombo_80split, zipcombo_20split = train_test_split(zipcombo, test_size=0.2, random_state=i)
        x_train_80 = zipcombo_80split[:, 1:]

        # TRAIN
        kernal_matrix_train_80 = _precompute_polykern_matrix(x1=x_train_80, x2=x_train_80, degree=degree)
        error_rate_prct_train, train_alpha = train_kp(ds=zipcombo_80split, k_mat=kernal_matrix_train_80, degree=degree,
                                                      k_classes=k_classes, epochs=epochs)
        error_rates_train[i] = error_rate_prct_train

        if write_results:
            with open(f'../saved_values/d{degree}/error_rate_prct_train_run#{i + 1}.txt', 'w') as f:
                f.write(str(error_rate_prct_train))
            np.savetxt(f'../saved_values/d{degree}/alpha_vec#{i+1}.csv', train_alpha)

        # pdf = pd.DataFrame(alpha_vec)
        # pdf.to_csv(f'saved_values/d{degree}/alpha_vec#{i+1}.csv', trained_alpha)

        # TEST
        x_test_20 = zipcombo_20split[:, 1:]
        kernel_matrix_train_80_test_20 = _precompute_polykern_matrix(x1=x_train_80, x2=x_test_20, degree=degree)
        error_rate_prct_test = test_kp(ds=zipcombo_20split, k_mat=kernel_matrix_train_80_test_20,
                                       trained_alpha=train_alpha)

        if write_results:
            with open(f'../saved_values/d{degree}/error_rate_prct_test_run#{i + 1}.txt', 'w') as f:
                f.write(str(error_rate_prct_train))

        error_rates_test[i] = error_rate_prct_test

    # AFTER ALL 20 RUNS, CALC MEANS & STDDEVS
    mean_train_error = np.mean(error_rates_train)
    stdev_train_error = np.std(error_rates_train)
    mean_test_error = np.mean(error_rates_test)
    stdev_test_error = np.std(error_rates_test)

    # PRINT AND SAVE VALUES
    print(f'mean train error={mean_train_error}, stdev train error={stdev_train_error}\n')
    print(f'mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')
    if write_results:
        with open(f'../saved_values/d{degree}/mean_stddev_train_error.txt', 'w') as f1:
            f1.write(f'degree {degree}, mean train error={mean_train_error}\nstdev train error={stdev_train_error}\n')
        with open(f'../saved_values/d{degree}/mean_stddev_test_error.txt', 'w') as f2:
            f2.write(f'degree {degree}, mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')


if __name__ == '__main__':

    start_time = time.time()
    zipcombo = np.loadtxt('../../datasets/zipcombo.dat')
    print(f'zipcombo.shape = {zipcombo.shape}')

    mini = zipcombo[:100, :]

    for i in range(7):
        degree = i + 1
        print(f'degree: {i + 1}')
        run_kp(zipcombo=zipcombo, degree=degree, num_of_runs=20, k_classes=10, epochs=5, write_results=True)

    print(f'time taken = {round(time.time() - start_time, 4)} seconds')

