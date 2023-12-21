# Iterate over number of classes (10 classes for the full MNIST dataset zipcombo.dat) j in the mathematica
# code, and iterating over number of examples (i in the mathematica), hence double for-loop.

# 1. Convert values of y in dataset, i.e. label , to +1 if they are the same as the class j,
# -1 otherwise.

# Weights are alpha in the equation (`GLBcls` in the mathematica).
# (Also described as a "classifier" and "an array of coefficients by Herbster).
# Learning occurs by updating the value of this.

# I'm not sure what to call w. It seems to be the summed weighted product.
# Multiply it by a data point, then do a signum thing, to get a prediction.

# As I iterate through each class and data point (a data point being 256 pixel values of an MNIST digit), I
# 2. make a prediction by multiplying each weight alpha_t, by the kernel evaluation.
# The kernel evaluation is an inner product of each of the other data points and the current data point).
# However, this inner product should only be done with the preceding data points, i.e. if the current data point is the
# 2nd one, I should only do the inner product with data point 0, and multiply it by alpha for 0,
# then add this to the product of data point 1 and multiply it by alpha for 1.
# So, I'm taking the sum of the weighted inner products up to the current data point only.
# This appears to be related to the "online learning" method.

# Finally, the prediction yt_hat is the output of the sign function, so the prediction is either +1 or -1.
# And given the true y labels were converted to +1 or -1, I just
# 3. check if yt_hat = yt. If it is, set weight (`alpha_t`) to 0, else to yt.

# I'm not clear if there is more to do for implementing final line of the algorithm table graphic, or if this is
# already covered by updating alpha..?  i.e. `w_t+1(.) = w_t(.) + a_t * K(xt, .)`

import numpy as np
import myfunctions as func
from tqdm import tqdm
import time

if __name__ == '__main__':

    start_time = time.time()

    zipcombo = np.loadtxt('../../datasets/zipcombo.dat')
    print(f'zipcombo shape = {zipcombo.shape}')
    m = len(zipcombo)
    X = zipcombo[:, 1:]
    y = zipcombo[:, 0]
    k_classes = 10
    alpha = np.zeros((k_classes, m))  # shape (10, 9298)
    degree = 3  # will need to run for degrees, from 1 to 7.
    mistakes = 0

    for k in tqdm(range(k_classes)):
        for t in range(m):
            # Convert y labels to +1 or -1, (as this is a one-vs-all binary classifier for each class):
            y[t] = 1.0 if y[t] == k else -1.0

            x_upto_t = X[:t]
            alpha_upto_t = alpha[k, :t]

            # Make prediction:
            yt_hat = func.predict_digit(x_upto_t=x_upto_t, xt=X[t], t=t, alpha_upto_t=alpha[k, :t], d=degree)
            alpha_kt = alpha[k, t]

            # Update alpha and count mistakes:
            if y[t] != yt_hat:
                alpha[k, t] = y[t]
                mistakes += 1

    print(f'mistakes = {mistakes}') # I got 256 mistakes on one run of full zipcombo.dat dataset. It took about 15
    # mins to run!!
    print(f'time taken = {time.time() - start_time}')

