import numpy as np
import some_funcs as func
import time
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import pandas as pd


def test_kp(ds, trained_alpha, d, k_classes):
    m = len(ds)
    X = ds[:, 1:]
    y = ds[:, 0]
    trained_alpha = np.zeros((k_classes, m))  # shape (10, 9298)
    mistakes = 0

    for k in range(k_classes):
        for t in range(m):
            y[t] = 1.0 if y[t] == k else -1.0

            # Make prediction:
            yt_hat = func.predict_digit(x_upto_t=X[:t], xt=X[t], t=t, alpha=trained_alpha[k, :], d=d)

            # Count mistakes, but make no updates:
            if y[t] != yt_hat:
                mistakes += 1
    print(f'mistakes during testing = {mistakes}')
    error_rate = mistakes / m
    return error_rate


def train_kp(ds, d, k_classes, epochs):
    """
    Perform a form of online learning by updating the weight (`alpha`) by updating its value the weighted kernel
    evaluation whenever its prediction is wrong.
    :param ds:
    :param d:
    :param k_classes:
    :param epochs:
    :return:
    """
    m = len(ds)
    alpha_vec = np.zeros((k_classes, len(ds)))
    X = ds[:, 1:]
    y = ds[:, 0]
    k_mat = func.precompute_polykern_matrix(x1=X, x2=X, degree=d)
    y_vec = func.convert_y_to_vector(y, k_classes)
    # y_vec_ep, ds_ep, alpha_ep, k_mat_ep = func.multiply_by_epochs(epochs, y_vec, ds, alpha_vec, k_mat)
    # m_ep = len(ds)
    mistakes = 0

    for epoch in range(epochs):
        # Make predictions for all classes:
        # preds = func.predict_digit_vec(alpha_vec, k_mat, k_classes, m)
        assert alpha_vec.shape[0] == y_vec.shape[0] == k_classes
        assert k_mat.shape[0] == k_mat.shape[1] == alpha_vec.shape[1] == m
        signs = np.ones((k_classes, 1))

        preds = np.zeros((m, k_classes))
        for t in range(m):
            a_slice = alpha_vec[:, :t]
            k_slice = k_mat[:t, t].reshape(-1, 1)
            pred = np.dot(a_slice, k_slice)
            preds[t] = pred.T
            prod_y_and_preds_at_t = y_vec[:, t] * preds.T[:, t]
            # Update any mis-classified
            # Maybe introduce a learning rate hyperparameter here ??
            # For example: lr = 0.2; alpha_ep = alpha_ep - (lr * func.sign(preds)) ??
            mask = prod_y_and_preds_at_t <= 0
            alpha_vals = alpha_vec[mask, t]
            alpha_vals = alpha_vals.reshape(-1, 1)
            preds_t = preds.T[:, t]
            signs[preds_t <= 0.0] = -1.0
            signs_masked = signs[mask].reshape(-1, 1)
            alpha_vec[mask, t] = (alpha_vals - signs_masked).flatten()
            # or ..?
            # alpha_vec[mask] = y_vec[mask]
        # Identifying mis-classifications:
        # In question sheet, mis-classifications are yt_hat != true yt (with yt_hat is calculated by sign(preds)).
        # In the mathematica code, mis-classifications are where preds * true yt is <= 0.
        # They achieve the same thing I think:
        # But, I'm not sure if in fact the following is the best way for this.
        # Find which digit was predicted to be most likely by finding the index of the highest pred score:

        preds = preds.T
        # most_likely_digits_predicted = np.argmax(preds, axis=0).reshape(1, -1)
        # assert most_likely_digits_predicted.shape[1] == preds.shape[1] == m
        # misclassified = most_likely_digits_predicted != y
        # misclassified_digits_actual = y[misclassified]
        # misclassified_digits_prediction = most_likely_digits_predicted[misclassified]
        # for i in range(len(misclassified_digits_actual)):
        #     print(f'{misclassified_digits_actual[i]} was misclassified as '
        #           f'{misclassified_digits_prediction[i]}')

        print(f'mistakes during training, epoch#{epoch} = {mistakes}')
    error_rate = mistakes / m
    return error_rate, alpha_vec


def run_kp(zipcombo, degree, num_of_runs, k_classes, epochs):

    trained_alpha_list = np.zeros(num_of_runs)
    train_mean_errors, test_mean_errors = np.zeros(num_of_runs), np.zeros(num_of_runs)

    for i in range(num_of_runs):  # serves dual purpose: loop n times, as well as providing seed for unique splits.

        zipcombo_80split, zipcombo_20split = train_test_split(zipcombo, test_size=0.2, random_state=i)
        error_rate, alpha_vec = train_kp(ds=zipcombo_80split, d=degree,
                                                               k_classes=k_classes, epochs=epochs)
        train_mean_errors[i] = error_rate
        # trained_alpha_list[i] = alpha_vec

        trained_alpha_path = f'saved_values/d{degree}/trained_alpha_run#{i+1}.csv'
        pdf = pd.DataFrame(alpha_vec)
        pdf.to_csv(trained_alpha_path)

        # test_mean_errors[i] = test_kp(ds=zipcombo_20split, trained_alpha=trained_alpha_list,
        #                               d=degree, k_classes=k_classes)

    mean_train_error = np.mean(train_mean_errors)
    stdev_train_error = np.std(train_mean_errors)
    # mean_test_error = np.mean(test_mean_errors)
    # stdev_test_error = np.std(test_mean_errors)
    print(f'mean train error={mean_train_error}, stdev train error={stdev_train_error}\n')
    # print(f'mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')

    with open(f'saved_values/d{degree}/train_mistakes_rate.txt', 'w') as f1:
        f1.write(f'mean train error={mean_train_error}\nstdev train error={stdev_train_error}\n')
    # with open(f'saved_values/d{degree}/test_mistakes_rate.txt', 'w') as f2:
    #     f2.write(f'mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')


if __name__ == '__main__':

    start_time = time.time()
    # Basic results
    # degrees to use are 1, 2, 3, 4, 5, 6, 7.
    zipcombo = np.loadtxt('../../datasets/zipcombo.dat')
    # ds_size = zipcombo.shape
    # mini_ = zipcombo[:1000, :]
    for deg in range(7):
        print(f'degree: {deg+1}')
        run_kp(zipcombo=zipcombo, degree=deg+1, num_of_runs=20, k_classes=10, epochs=3)

    print(f'time taken = {round(time.time() - start_time, 4)} seconds')

