#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:09:14 2021

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
from train_no_constraint import net_evaluate

# init 
_seed = 101
rng = jax.random.PRNGKey(_seed)

curr_path = os.path.dirname(__file__)
data_path = os.path.join(curr_path, f"data_seed_{_seed}.pkl")
params_path_xy = os.path.join(curr_path, f"baseline_xy_params_seed_{_seed}.pkl")
params_path_yx = os.path.join(curr_path, f"baseline_yx_params_seed_{_seed}.pkl")

with open(params_path_xy, mode='rb') as fp:
    params_xy = pickle.load(fp)
with open(params_path_yx, mode='rb') as fp:
    params_yx = pickle.load(fp)
    
def plot_scatter(X,Y, X_hat, Y_hat):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(X, Y, facecolor='none', edgecolors='k', alpha=0.5)
    ax1.plot(X, Y_hat, c='r')
    ax1.set_title(r'Forward mapping $f\colon x \mapsto f(x) = y$', fontsize=8)
    ax1.set_xlabel("x values")
    ax1.set_ylabel("y values")
    idx = np.argsort(Y.ravel())
    ax2.plot(Y.ravel()[idx], X_hat.ravel()[idx], c='b')
    ax2.scatter(Y, X, facecolor='none', edgecolors='k', alpha=0.5)
    ax2.set_title(r'Backward mapping $g\colon y \mapsto g(y) = x$', fontsize=8)
    ax2.set_xlabel("y values")
    ax2.set_ylabel("x values")
    plt.tight_layout()
    plt.show()
    
    

if __name__ == '__main__':
    # load data
    data = pd.read_pickle(data_path)
    X = data['x'].values.reshape(-1,1)
    Y = data['y'].values.reshape(-1,1)
    # init net & opt
    net = hk.without_apply_rng(hk.transform(net_evaluate))
    # get result of evaluate
    Y_hat = net.apply(params_xy, X,Y)
    X_hat = net.apply(params_yx, Y,X)
    
    plot_scatter(X,Y, X_hat, Y_hat)