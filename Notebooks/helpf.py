import scipy as sp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from statsmodels.formula.api import ols
from scipy.stats import linregress, pearsonr
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Lasso, Ridge, LinearRegression


def qqplot(x, y, data):
    lr_model = ols(formula=f"{y}~{x}", data=data).fit()
    pred_val = lr_model.fittedvalues.copy()
    true_val = data[f'{y}'].values.copy()
    residual = true_val - pred_val
    fig, ax = plt.subplots(figsize=(6, 2.5))
    sp.stats.probplot(residual, plot=ax, fit=True)


def scater(x, y, data):
    sns.jointplot(data[f'{x}'], data[f'{y}'], kind='reg')


def regression(x, y, color):
    reg = linregress(x, y)
    corr, _ = pearsonr(x, y)
    if corr > 0:
        print('Positive correlation')
    elif corr < 0:
        print('Negative correlation')
    else:
        print('No correlation')
    print('Pearsons correlation: %.3f' % corr)
    print(reg)
    sns.regplot(x, y, color=color)


def split_test(X, y, size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    train = len(y_train)
    test = len(y_test)
    print(f"training set has {train} observations")
    print(f"testing set has {test} observations")
    print('\n')

    ridge = Ridge(alpha=0.5)
    ridge.fit(X_train, y_train)

    lasso = Lasso(alpha=0.5)
    lasso.fit(X_train, y_train)

    lin = LinearRegression()
    lin.fit(X_train, y_train)

    y_h_ridge_train = ridge.predict(X_train)
    y_h_ridge_test = ridge.predict(X_test)

    y_h_lasso_train = lasso.predict(X_train)
    y_h_lasso_test = lasso.predict(X_test)

    y_h_lin_train = lin.predict(X_train)
    y_h_lin_test = lin.predict(X_test)

    print('Train Error Ridge Model', np.sum((y_train - y_h_ridge_train)**2))
    print('Test Error Ridge Model', np.sum((y_test - y_h_ridge_test)**2))
    print('\n')

    print('Train Error Lasso Model', np.sum((y_train - y_h_lasso_train)**2))
    print('Test Error Lasso Model', np.sum((y_test - y_h_lasso_test)**2))
    print('\n')

    print('Train Error Unpenalized Linear Model', np.sum((y_train - lin.predict(X_train))**2))
    print('Test Error Unpenalized Linear Model', np.sum((y_test - lin.predict(X_test))**2))


def cross_validation(X, y):
    ridge = linear_model.Ridge()
    cv = cross_validate(ridge, X, y, scoring=(
        'r2', 'neg_mean_squared_error'), return_train_score=True)
    test2 = list(cv['test_r2'])
    train2 = list(cv['train_r2'])
    testmse = list(cv["test_neg_mean_squared_error"])
    trainmse = list(cv["train_neg_mean_squared_error"])

    print(f"Test r2: {test2}")
    print(f"Train r2: {train2}")
    print(f"Test MSE: {testmse}")
    print(f"Train MSE: {trainmse}")
