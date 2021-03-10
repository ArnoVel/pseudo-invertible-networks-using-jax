#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:59:12 2021

@author: arnovel
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import haiku as hk
import jax

# local imports
from train_sample_constraint import net_evaluate

# init 
_seed = 101
rng = jax.random.PRNGKey(_seed)

curr_path = os.path.dirname(__file__)
data_path = os.path.join(curr_path, f"data_seed_{_seed}.pkl")
params_path = os.path.join(curr_path, f"params_seed_{_seed}.pkl")

with open(params_path, mode='rb') as fp:
    params = pickle.load(fp)
    

if __name__ == '__main__':
    # load data
    data = pd.read_pickle(data_path)
    X = data['x'].values.reshape(-1,1)
    Y = data['y'].values.reshape(-1,1)
    # init net & opt
    net = hk.without_apply_rng(hk.transform(net_evaluate))
    # get result of evaluate
    X_hat, Y_hat, inv_X, inv_Y = net.apply(params, X,Y)
    plt.scatter(X, Y, facecolor='none', edgecolors='k', alpha=0.5)
    plt.plot(X, Y_hat, c='r')
    plt.title(r'Forward mapping $f\colon x \mapsto f(x) = y$', fontsize=15)
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.show()
    idx = np.argsort(Y.ravel())
    plt.plot(Y.ravel()[idx], X_hat.ravel()[idx], c='b')
    plt.scatter(Y, X, facecolor='none', edgecolors='k', alpha=0.5)
    plt.title(r'Backward mapping $g\colon y \mapsto g(y) = x$', fontsize=15)
    plt.xlabel("y values")
    plt.ylabel("x values")
    plt.show()

    # plt.legend()
    