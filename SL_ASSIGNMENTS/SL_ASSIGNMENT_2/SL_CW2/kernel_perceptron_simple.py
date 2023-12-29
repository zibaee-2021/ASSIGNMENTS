# python 3.10.13, in Miniconda environment,
# numpy 1.26.2, numba 0.58.1, matplotlib 3.8.2, pandas 2.1.4.

import numpy as np
import myfunctions as func
import time
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import pandas as pd


def test_kp(test_split, trained_alpha, d, k_classes):
    m = len(test_split)
    X = test_split[:, 1:]
    y = test_split[:, 0]
    trained_alpha = np.zeros((k_classes, m))  # shape (10, 9298)
    mistakes = 0

    for k in range(k_classes):
        for t in range(m):
            y[t] = 1.0 if y[t] == k else -1.0

            # Make prediction:
            yt_hat = func.predict_digit(x_upto_t=X[:t], xt=X[t], t=t, alpha_k=trained_alpha[k, :], d=d)

            # Count mistakes, but make no updates:
            if y[t] != yt_hat:
                mistakes += 1
    print(f'mistakes during testing = {mistakes}')
    error_rate = mistakes / m
    return error_rate


def train_kp(train_split, d, k_classes):
    print(f'train_split.shape {train_split.shape}')
    alpha = np.zeros((k_classes, len(train_split)))
    epochs = 3
    # Increase size of dataset according to number of epochs
    # Note: I leave alpha at its original size and not the size of the dataset after expansion for epochs
    train_split = np.tile(train_split, (epochs, 1))
    X = train_split[:, 1:]
    y = train_split[:, 0]

    m = len(train_split)
    mistakes = 0

    for k in tqdm(range(k_classes)):
        for t in range(m):
            # print(f't={t} of {m}')
            # Convert y labels to +1 or -1, (as this is a one-vs-all binary classifier for each class):
            y[t] = 1.0 if y[t] == k+1 else -1.0

            # Make prediction:
            yt_hat = func.predict_digit(x_upto_t=X[:t], xt=X[t], t=t, alpha_k=alpha[k, :], d=d)

            # Update alpha and count mistakes:
            if y[t] != yt_hat:
                # Maybe introduce a learning rate hyperparameter here ??
                # For example: lr = 0.2; alpha[k, t] = alpha[k, t] + lr * y[t]; ??

                t_for_alpha = t % len(alpha)
                alpha[k, t_for_alpha] = alpha[k, t_for_alpha] + y[t]
                mistakes += 1
    print(f'mistakes during training = {mistakes}')
    error_rate = mistakes / m

    return error_rate, alpha


def train_and_test_kp(zipcombo, degree, num_of_runs=20, k_classes=10):

    train_mean_errors, test_mean_errors = np.zeros(num_of_runs), np.zeros(num_of_runs)

    for i in range(num_of_runs):  # serves dual purpose: loop 20 times, as well as providing seed for unique splits.

        zipcombo_80split, zipcombo_20split = train_test_split(zipcombo, test_size=0.2, random_state=i)
        # shape (10, 9298)
        train_mean_errors[i], trained_alpha = train_kp(train_split=zipcombo_80split,
                                                       d=degree, k_classes=k_classes)
        trained_alpha_path = f'saved_values/d{degree}/trained_alpha.csv'
        pdf = pd.DataFrame(trained_alpha)
        pdf.to_csv(trained_alpha_path)

        test_mean_errors[i] = test_kp(test_split=zipcombo_20split, trained_alpha=trained_alpha,
                                      d=degree, k_classes=k_classes)

    mean_train_error = np.mean(train_mean_errors)
    stdev_train_error = np.std(train_mean_errors)
    mean_test_error = np.mean(test_mean_errors)
    stdev_test_error = np.std(test_mean_errors)
    print(f'mean train error={mean_train_error}, stdev train error={stdev_train_error}\n')
    print(f'mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')

    with open(f'saved_values/d{degree}/train_mistakes_rate.txt', 'w') as f1:
        f1.write(f'mean train error={mean_train_error}\nstdev train error={stdev_train_error}\n')
    with open(f'saved_values/d{degree}/test_mistakes_rate.txt', 'w') as f2:
        f2.write(f'mean test error={mean_test_error}\nstdev test error={stdev_test_error}\n')


if __name__ == '__main__':

    start_time = time.time()
    # Basic results
    # degrees to use are 1, 2, 3, 4, 5, 6, 7.
    train_and_test_kp(zipcombo=np.loadtxt('../datasets/zipcombo.dat'), degree=1,
                      num_of_runs=2, k_classes=10)

    # train_and_test_kernel_perceptron(zipcombo=np.loadtxt('../../datasets/dtrain123.dat'), degree=1,
    #                                  num_of_runs=2, k_classes=3)
    print(f'time taken = {time.time() - start_time}')

