#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:27:23 2021

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
## data path
curr_path = os.path.dirname(__file__)
data_path = os.path.join(curr_path, f"data_seed_{_seed}.pkl")
params_path = os.path.join(curr_path, f"baseline_xy_params_seed_{_seed}.pkl")

# funcs

def two_layers_net(width: int = 30,
                   output_dim: int = 1
                   ) -> hk.Module:
    '''
    A basic two layer network with ReLU activations
    '''
    network = hk.Sequential([
        hk.Linear(width), jax.nn.silu,
        hk.Linear(width), jax.nn.silu,
        hk.Linear(output_dim)
        ])
    
    return network

def net_evaluate(X: array,
                 Y: array,
                 width: int = 30,
                 ) -> array:
    '''
    Evaluates the two networkx on data `X`, `Y`.
    '''
    output_dim = Y.shape[1]
    network = two_layers_net(width, output_dim)
    
    Y_hat = network(X)
    
    return Y_hat
    

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
             ) -> array:
    
        Y_hat = net.apply(params, X, Y)
        
        avg_y_fit_sq_err = jnp.mean( (Y_hat - Y) ** 2 )
                
        _loss = avg_y_fit_sq_err
        
        return _loss
    
    @jit
    def update(params: hk.Params,
               opt_state: OptState,
               X: array,
               Y: array,
               ) -> Tuple[hk.Params, OptState]:
        
        grads = grad(loss)(params, X, Y)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    
    hist_train = []
    # train for `max_iters` epochs
    for step in tqdm(range(max_iters)):
        _loss = loss(params, X, Y)
        hist_train.append(_loss)
        params, opt_state = update(params, opt_state, X, Y)
        
        
    # plot train error as func(iter)
    plt.plot(hist_train)
    plt.show()
    
    # store params
    with open(params_path, mode='wb') as fp:
        pickle.dump(obj=params, file=fp)