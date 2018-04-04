#coding=utf-8

import numpy as np
import pandas as pd

def h_N(h1, N):
    return h1/np.sqrt(N)

def window_func_rec(u):
    if u.max() <= 0.5 and u.max() >= -0.5:
        return 1
    return 0

def window_func_norm(u):
    return 1/np.sqrt(2*np.pi)*np.exp(-0.5* u.dot(u[:,None]))

def window_func_exp(u):
    return np.exp(-0.5*np.linalg.norm(u, ord=1))/4

def prob(data, x, h, N, V, window_func):
    data["window"] = data.apply(lambda xi: window_func((x-xi[0])/h), axis=1)
    return data["window"].sum()/(N*V)

def estimate(data, plots, h1, interval_len):
    N = data.size
    d = data[0].size
    h = h_N(h1, N)
    V = h**d
    window_func = window_func_exp
    data = pd.DataFrame(data, columns=["data"])
    prediction = pd.DataFrame(plots, columns=["x"])
    prediction["prob"] = prediction.apply(lambda x : prob(data, x, h, N, V, window_func), axis=1)
    prediction["interval_prob"] = prediction.apply(lambda x : x[1]*interval_len, axis=1)

    return prediction