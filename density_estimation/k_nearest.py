#coding=utf-8

import numpy as np
import pandas as pd

def k_N(k1, N):
    return int(np.ceil(np.sqrt(N))/k1)

def find_k_nearest(data, x, k):
    data["dist"]=data.apply(lambda xi: np.linalg.norm(x-xi[0], ord=np.inf), axis=1)
    data = data.sort_values(by="dist")
    return data.iloc[k-1]["dist"]

def kernel_uniform(u):
    if u.max() <= 0.5 and u.max() >= -0.5:
        return 1.0
    return 0

def kernel_norm(u):
    sigma = 1
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-u**2/(2*sigma))

def kernel_exp(u):
    return np.exp(-0.5*u)/4

def calc_prob(N, d, data):
    kernel = kernel_exp
    data["kernel"] = data.apply(lambda xi: kernel(xi[1]/d), axis=1)
    return 1/(N*d)*data["kernel"].sum()

def estimate(data, plots, k1, interval_len):
    N = data.size
    d = data[0].size
    k = k_N(k1, N)
    data = pd.DataFrame(data, columns=["data"])
    prediction = pd.DataFrame(plots, columns=["x"])
    prediction["prob"] = prediction.apply(lambda x: calc_prob(N, find_k_nearest(data, x, k), data), axis=1)
    prediction["interval_prob"] = prediction.apply(lambda x: x[1] * interval_len, axis=1)

    return prediction