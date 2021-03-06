#coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pazzle
import k_nearest

lowerbound = -4.0
upperbound = 4.0
size = 80
interval_len = (upperbound - lowerbound) / size
plots = pd.DataFrame(np.arange(lowerbound + interval_len / 2, upperbound, interval_len), columns=["x"])

def generalize_data_normal(size):
    mu = 0
    sigma = 1
    data = np.random.normal(mu, sigma, size);
    plots["real_prob"] = plots.apply(lambda x:1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x[0] - mu)**2 / (2 * sigma**2)), axis=1)
    return data

def estimate(data, h):
    return pazzle.estimate(data, plots["x"], h, interval_len)

def draw(prediction, train_size, h, figure):
    x = prediction["x"]
    p = prediction["prob"]
    real_p = plots["real_prob"]
    figure.plot(x, real_p, color="r")
    figure.plot(x, p, color="b")
    max_y = max(real_p.max(), p.max())
    figure.text(lowerbound, max_y/2, "N=%d\nh1=%f" % (train_size, h))
    return

if __name__ == "__main__":
    data_size = 5000
    generalize_data = generalize_data_normal
    data = generalize_data(data_size);
    rows = 3
    cols = 4
    i = 1

    arg = [[(20,0.1),(20,1),(20,5),(20,10)],
           [(100,0.1),(100,1),(100,5),(100,10)],
           [(500,0.1),(500,1),(500,5),(500,10)]
    ]

    for a in arg:
        for train_size,h1 in a:
            train_data = np.random.choice(data, train_size, replace=False)
            k = h1
            prediction = estimate(train_data, k)
            print(k)
            draw(prediction, train_size, k, plt.subplot(rows, cols, i))
            i += 1
    plt.show()
