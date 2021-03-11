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
    
def plot_inverse(X,Y, inv_X, inv_Y):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(X, inv_X, c='r', alpha=0.5)
    ax1.plot(X, X, linestyle='--', color='k', alpha=0.3)
    ax1.set_title(r'Composition $x\colon x \mapsto g\circ f(x)$', fontsize=8)
    ax1.set_xlabel("x values")
    ax1.set_ylabel(r"$g\circ f(x)$ values")
    idx = np.argsort(Y.ravel())
    ax2.plot(Y.ravel()[idx], inv_Y.ravel()[idx], c='b', alpha=0.5)
    ax2.plot(Y, Y, linestyle='--', color='k', alpha=0.3)
    ax2.set_title(r'Composition $y\colon y \mapsto f\circ g(y)$', fontsize=8)
    ax2.set_xlabel("y values")
    ax2.set_ylabel(r"$f\circ g(y)$ values")
    plt.tight_layout()
    plt.show()
    
def plot_scatter_inverse(X,Y, X_hat, Y_hat, inv_X, inv_Y):
    fig, axes = plt.subplots(2, 2, figsize=(15,15))
    axes[0,0].scatter(X, Y, facecolor='none', edgecolors='k', alpha=0.5)
    axes[0,0].plot(X, Y_hat, c='r')
    axes[0,0].set_title(r'Forward mapping $f\colon x \mapsto f(x) = y$', fontsize=8)
    axes[0,0].set_xlabel("x values")
    axes[0,0].set_ylabel("y values")
    idx = np.argsort(Y.ravel())
    axes[0,1].plot(Y.ravel()[idx], X_hat.ravel()[idx], c='b')
    axes[0,1].scatter(Y, X, facecolor='none', edgecolors='k', alpha=0.5)
    axes[0,1].set_title(r'Backward mapping $g\colon y \mapsto g(y) = x$', fontsize=8)
    axes[0,1].set_xlabel("y values")
    axes[0,1].set_ylabel("x values")
    axes[1,0].plot(X, inv_X, c='r', alpha=0.5)
    axes[1,0].plot(X, X, linestyle='--', color='k', alpha=0.3)
    axes[1,0].set_title(r'Composition $x\colon x \mapsto g\circ f(x)$', fontsize=8)
    axes[1,0].set_xlabel("x values")
    axes[1,0].set_ylabel(r"$g\circ f(x)$ values")
    idx = np.argsort(Y.ravel())
    axes[1,1].plot(Y.ravel()[idx], inv_Y.ravel()[idx], c='b', alpha=0.5)
    axes[1,1].plot(Y, Y, linestyle='--', color='k', alpha=0.3)
    axes[1,1].set_title(r'Composition $y\colon y \mapsto f\circ g(y)$', fontsize=8)
    axes[1,1].set_xlabel("y values")
    axes[1,1].set_ylabel(r"$f\circ g(y)$ values")
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
    X_hat, Y_hat, inv_X, inv_Y = net.apply(params, X,Y)
    
    # plot_scatter(X,Y, X_hat, Y_hat)
    # plot_inverse(X, Y, inv_X, inv_Y)
    plot_scatter_inverse(X,Y, X_hat, Y_hat, inv_X, inv_Y)
    # plt.legend()
    