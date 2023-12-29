import numpy as np
import time
import os
import matplotlib.pyplot as plt
import shared_functions as shfun


def _train_kp_only(y, k_mat, k_classes: int):
    """
    Train weights but do not record mistake count here.
    :param y: Labels for training dataset split of MNIST. As 1d array with shape (m, 257) where `m` is number of
    datapoints (epoch-tiled).
    :param k_mat: Kernel matrix for given training dataset (epoch-tiled).
    :param k_classes: Number of classes for which to perform classification.
    :return: The error rate percentage and the trained weights for the given dataset, degree, k_classes and epochs.
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
        learning_rate = 1
        y_vec_with_which_to_update_alpha = learning_rate * y_vec[misclass_mask_at_t, t].reshape(-1)
        alpha_vec[misclass_mask_at_t, t] = y_vec_with_which_to_update_alpha

    return alpha_vec


# -------- Q4. USING FULL DATASET, TRAIN WEIGHTS WITH MEAN d* OVER 20 RUNS. ------------------------------------------
# ------- THEN PERFORM PREDICTIONS ON FULL DATASET AND RECORD EACH MIS-CLASSIFICATION OVER 20 RUNS: ------------------


def train_weights_and_store_misclassifications(train_new_weights, ds, d_star, num_of_runs=20, k_classes=10, epochs=3):
    """
    Train weights (`alpha`) on full dataset for given number of epochs and given nymber of independent runs.
    :param train_new_weights: True to train weights. False to read in pre-trained weights from file.
    :param ds: Dataset. Expected to be full MNIST dataset. 2d array (m, 257).
    :param d_star: Mean of 20 best degrees for polynomial kernel evaluation (calculated in question 2).
    :param num_of_runs: Number of independent runs for making predictions only. 20 by default.
    :param k_classes: Number of classes to perform classification over. 10 by default.
    :param epochs: Number of epochs to train the weights over. 3 by default.
    :return: All labels and misclassified predictions or -1. As 2d array of shape (m, 2)
    """
# -- 1st: GET THE WEIGHTS FOR WHOLE DATASET AND `d_star` FROM QUESTION 2 (i.e. 5.55) --------------------------

    # GET DATA POINTS AND LABELS FROM FULL DATASET SET:
    x = ds[:, 1:]
    y = ds[:, 0]

    # PRE-COMPUTE KERNEL MATRIX:
    kernel_matrix = shfun.compute_polykern_matrix(x1=x, x2=x, degree=d_star)

    # MODEL ADDITIONAL EPOCHS BY EXTENDING THE DATA BY MULTIPLICATION:
    y = np.tile(A=y, reps=epochs)
    k_mat = np.tile(A=kernel_matrix, reps=(epochs, epochs))

    # TRAIN WEIGHTS ONLY (NOT COUNTING MISTAKES YET):
    if train_new_weights:

        # TRAIN WEIGHTS ONLY (NOT COUNTING MISTAKES YET):
        trained_alpha = _train_kp_only(y=y, k_mat=k_mat, k_classes=k_classes)

        # WRITE WEIGHTS TO FILE (INCASE YOU NEED THEM AGAIN AND DON'T WANT TO REPEAT THIS STEP):
        dir = f'../saved_values'
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir = f'../saved_values/Q4'
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.savetxt('../saved_values/Q4/trained_alpha.txt', trained_alpha.astype(np.int32), fmt='%d')

    else:
        trained_alpha = np.loadtxt('../saved_values/Q4/trained_alpha.txt', dtype=int)


# -- 2nd: MAKE PREDICTIONS WITH TRAINED WEIGHTS, USING WHOLE DATASET AND PRE-COMPUTED KERNEL MATRIX FROM ABOVE -------

    # PREDICT
    predictions_with_trained_alpha = np.dot(trained_alpha, k_mat)

    # Remove the additional epochs as data is just replicated:
    predictions_with_trained_alpha = predictions_with_trained_alpha[:, :len(ds)]
    y = y[:len(ds)]

    # digit & index happen to be same number for 10 classes with 0-based indexing.
    predicted_y = np.argmax(predictions_with_trained_alpha, axis=0).reshape(-1, 1)
    true_y = y.reshape(-1, 1).astype(np.int32)
    true_and_preds = np.concatenate([true_y, predicted_y], axis=1)  # just for viewing
    misclass_mask = predicted_y != true_y
    misclassed_or_neg_one = np.where(misclass_mask, predicted_y, -1)
    misclass_list = np.concatenate([true_y, misclassed_or_neg_one], axis=1)

    # WRITE MIS-CLASSIFICATIONS TO FILE
    dir = f'../saved_values'
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = f'../saved_values/Q4'
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.savetxt('../saved_values/Q4/misclass_list.txt', misclass_list, fmt='%d')

    return misclass_list


def _plot_mnist_digit(label_and_pixels):
    """
    Plot the given 256 grayscale values to 16 x 16 image. First column is the label.
    :param label_and_pixels: 1st column: MNIST digit label, 2nd column: wrong prediction. Remaining 256 columns are
    pixels values for each MNIST digit.
    """
    assert len(label_and_pixels) == 258, 'Length of given array should be 257'
    pxl_img = np.array(label_and_pixels[2:]).reshape((16, 16))
    plt.imshow(pxl_img, cmap='gray')
    plt.title(f"{label_and_pixels[0].astype(np.int32)} predicted as {label_and_pixels[1]}")
    plt.show()


def get_misclassified_and_plot(ds):
    """
    Make list of mis-classified digits from file generated by `train_weights_and_store_misclassifications()`
    :param ds:Full dataset including pixel values of digits.
    """
    ds = np.loadtxt('../datasets/zipcombo.dat')
    misclasses = np.loadtxt('saved_values/Q4/misclass_list.txt', dtype=int)

    misclass_mask = misclasses != -1
    misclass_rows_only = misclasses[misclass_mask.all(axis=1)]
    wrong_predictions_only = misclass_rows_only[:, 1]

    misclass_mask = misclasses[:, 1] != -1
    misclassed_ds = ds[misclass_mask]

    misclassed_ds = np.insert(misclassed_ds, 1, wrong_predictions_only, axis=1)

    num_rows = 4
    num_cols = 5

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    axes = axes.flatten()

    for i, misclassed in enumerate(misclassed_ds):
        pxl_img = np.array(misclassed[2:]).reshape((16, 16))
        ax = axes[i]
        ax.imshow(pxl_img, cmap='gray')
        ax.axis('off')
        misclass_true = misclassed[0].astype(np.int32)
        misclass_pred = misclassed[1].astype(np.int32)
        ax.set_title(f"{misclass_true} predicted as {misclass_pred}", fontsize=10, pad=5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # _visualise_confusion_matrices()
    # mean_confs = np.loadtxt('../saved_values/mean_confs.txt')
    # mean_confs_sums = np.sum(mean_confs, axis=1)
    start_time = time.time()
    ds = np.loadtxt('../../datasets/zipcombo.dat')
    # ds = ds[:100, :]
    print(f'ds.shape = {ds.shape}')
    # misclasses = train_weights_and_store_misclassifications(train_new_weights=False, ds=ds, d_star=5,
    #                                                         num_of_runs=20, k_classes=10, epochs=3))
    get_misclassified_and_plot(ds)

    mins = (time.time() - start_time) / 60
    print(f'time taken = {round(mins, 4)} minutes')
