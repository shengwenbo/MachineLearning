#coding=utf-8

import numpy as np
import pandas as pd

def k_N(k1, N):
    return int(np.ceil(np.sqrt(N))/k1)

def find_k_nearest(data, x, k):
    data["dist"]=data.apply(lambda xi: np.linalg.norm(x-xi[0], ord=np.inf), axis=1)
    data = data.sort_values(by="dist")
    return data.iloc[k-1]["dist"]

def calc_V(h, d):
    return h**d

def calc_prob(k, N, V):
    return k/N/V

def estimate(data, plots, k1, interval_len):
    N = data.size
    d = data[0].size
    k = k_N(k1, N)
    data = pd.DataFrame(data, columns=["data"])
    prediction = pd.DataFrame(plots, columns=["x"])
    prediction["prob"] = prediction.apply(lambda x: calc_prob(k, N, calc_V(find_k_nearest(data, x, k), d)), axis=1)
    prediction["interval_prob"] = prediction.apply(lambda x: x[1] * interval_len, axis=1)

    return prediction