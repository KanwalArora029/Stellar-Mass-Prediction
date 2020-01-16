#import scipy as sp
import pandas as pd
import numpy as np


def qqplot(x, y, data):
    lr_model = ols(formula=f"{y}~{x}", data=data).fit()
    pred_val = lr_model.fittedvalues.copy()
    true_val = data[f'{y}'].values.copy()
    residual = true_val - pred_val
    fig, ax = plt.subplots(figsize=(6, 2.5))
    sp.stats.probplot(residual, plot=ax, fit=True)


def scater(x, y, data):
    sns.jointplot(data[f'{x}'], data[f'{y}'], kind='reg')
