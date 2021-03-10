#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 07:36:39 2021

@author: arnovel
"""
import os
import pickle
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import haiku as hk
import jax
import optax

from tqdm import tqdm
from jax import grad, jit
from jax.interpreters.xla import _DeviceArray
from typing import TypeVar, Any, Tuple

# types

OptState = Any
array = TypeVar("array", bound=_DeviceArray)


# init
_seed = 101
rng = jax.random.PRNGKey(_seed)
## number of Gradient Descent updates
max_iters = int(5e02)
## Gradient Descent Step Size
step_size = 1e-03
_lambda, _mu = 1.0, 1.0
## data path
curr_path = os.path.dirname(__file__)
data_path = os.path.join(curr_path, f"data_seed_{_seed}.pkl")
params_path = os.path.join(curr_path, f"params_seed_{_seed}.pkl")

# funcs

def two_layers_net(width: int = 30,
                   output_dim: int = 1
                   ) -> hk.Module:
    '''
    A basic two layer network with ReLU activations
    '''
    network = hk.Sequential([
        hk.Linear(width), jax.nn.relu,
        hk.Linear(width), jax.nn.relu,
        hk.Linear(output_dim)
        ])
    
    return network

def net_evaluate(X: array,
                 Y: array,
                 width: int = 30,
                 ) -> Tuple[array]:
    '''
    Evaluates the two networkx on data `X`, `Y`.
    '''
    output_dim = Y.shape[1]
    net_frwd = two_layers_net(width, output_dim)
    net_bkwd = two_layers_net(width, output_dim)
    
    Y_hat = net_frwd(X)
    X_hat = net_bkwd(Y)
    inv_X = net_bkwd(Y_hat)
    inv_Y = net_frwd(X_hat)
    
    return X_hat, Y_hat, inv_X, inv_Y
    

if __name__ == '__main__':
    # load data
    data = pd.read_pickle(data_path)
    X = data['x'].values.reshape(-1,1)
    Y = data['y'].values.reshape(-1,1)
    # init net & opt
    net = hk.without_apply_rng(hk.transform(net_evaluate))
    opt = optax.adam(step_size)
    params = net.init(rng, X, Y)
    opt_state = opt.init(params)
    
    # define runtime routines
    
    def loss(params: hk.Params,
             X: array,
             Y: array,
             _lambda: float,
             _mu: float,
             ) -> array:
    
        X_hat, Y_hat, inv_X, inv_Y = net.apply(params, X,Y)
        
        avg_x_fit_sq_err = jnp.mean( (X_hat - X) ** 2 )
        avg_y_fit_sq_err = jnp.mean( (Y_hat - Y) ** 2 )
        avg_x_invfit_sq_err = jnp.mean( (inv_X - X) ** 2 )
        avg_y_invfit_sq_err = jnp.mean( (inv_Y - Y) ** 2 )
        
        _fit = avg_x_fit_sq_err + avg_y_fit_sq_err
        _inv_constraint = _lambda * avg_x_invfit_sq_err + _mu * avg_y_invfit_sq_err
        
        _loss = _fit + _inv_constraint
        
        return _loss
    
    @jit
    def update(params: hk.Params,
               opt_state: OptState,
               X: array,
               Y: array,
               ) -> Tuple[hk.Params, OptState]:
        
        grads = grad(loss)(params, X, Y, _lambda, _mu)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    
    hist_train = []
    # train for `max_iters` epochs
    for step in tqdm(range(max_iters)):
        _loss = loss(params, X, Y, _lambda, _mu)
        hist_train.append(_loss)
        params, opt_state = update(params, opt_state, X, Y)
        
        
    # plot train error as func(iter)
    plt.plot(hist_train)
    plt.show()
    
    # store params
    with open(params_path, mode='wb') as fp:
        pickle.dump(obj=params, file=fp)
    