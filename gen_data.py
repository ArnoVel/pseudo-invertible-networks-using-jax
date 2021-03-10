#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 08:06:10 2021

@author: arnovel
"""

import os
import numpy as np
import pandas as pd
import scipy.interpolate as itp
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial 

# init
curr_path = os.path.dirname(__file__)
_seed = 101
np.random.seed(_seed)

def fit_random_spline(n_knots=10):
    y_knots = np.random.randn(n_knots)*1.5
    x_knots = np.linspace(-10, 10, n_knots)
    idx = np.argsort(x_knots)
    x_knots, y_knots = x_knots[idx], y_knots[idx]
    spl = itp.UnivariateSpline(x_knots, y_knots)
    return spl

def gaussian_spline_post_nonlinear(x, n_knots=10):
    phi = fit_random_spline(n_knots)
    psi = fit_random_spline(n_knots)
    eps = np.random.randn(len(x))*0.3
    
    return psi(phi(x) + eps)

xgrid = np.linspace(-10,10, 1000)
ygrid = gaussian_spline_post_nonlinear(xgrid)

# least squares polyfit
fpol = Polynomial.fit(xgrid, ygrid, deg=5)
ypol = fpol(xgrid)

plt.scatter(xgrid, ygrid, facecolor='none', edgecolors='k', alpha=0.5)
plt.plot(xgrid, ypol, label='poly', c='r')
plt.legend()
plt.show()

df = pd.DataFrame({'x':xgrid, 'y':ygrid})
data_path = os.path.join(curr_path, f"data_seed_{_seed}.pkl")
df.to_pickle(data_path)