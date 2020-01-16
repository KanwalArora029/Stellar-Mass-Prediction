import helpf as hf
import sys
from sklearn.preprocessing import Imputer
from scipy.stats import linregress
from sklearn.preprocessing import PolynomialFeatures
import scipy as sp
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
plt.style.use('seaborn')

# some_file.py
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '/Users/flatironschool/Documents/flatiron/Mod4_project/Notebooks')


# read csv
df = pd.read_csv(
    "/Users/flatironschool/Documents/flatiron/Mod4_project/Data/planets_stars.csv", error_bad_lines=False)
df.isna().sum()

new_df = df[np.isfinite(df["pl_trandur"])]
new_df = new_df[np.isfinite(new_df["pl_orbsmax"])]
new_df.shape
new_df = new_df[np.isfinite(new_df["st_mass"])]
new_df = new_df[np.isfinite(new_df["st_rad"])]
new_df = new_df[np.isfinite(new_df["st_teff"])]
new_df = new_df[np.isfinite(new_df["pl_radj"])]
new_df.shape
new_df.isna().sum()
new_df.head(2)
X = new_df[["pl_trandur", "pl_orbsmax"]]

Y = new_df['st_mass'].apply(lambda x: x*1.989e+30)
X['pl_trandur_metric'] = X['pl_trandur'].apply(lambda x: x*8.64e+4)
X['pl_orbsmax_metric'] = X['pl_orbsmax'].apply(lambda x: x*1.496e+8)


ln_df = new_df[["st_mass", "pl_trandur", "pl_orbsmax"]]

X.columns
cols = ['pl_trandur_metric', 'pl_orbsmax_metric']

columns_log(cols, X)

X.columns


X_metric = X[['log_pl_trandur_metric', 'log_pl_orbsmax_metric']]
Y_metric = np.log(Y)

#X_metric = sm.add_constant(X_metric)


model_metric = sm.OLS(Y_metric, X_metric).fit()

model_metric.summary()
model_metric.params


cols

hf.columns_zc(cols, X)

poli_features = PolynomialFeatures(degree=2)
Xp = poli_features.fit_transform(X_rad_zcore)
poli_features.get_feature_names(X_rad_zcore.columns)
Xp.shape
Xp[:, 6:]
model = sm.OLS(new_df.st_mass, Xp[:, [0, 1, 2, 3, 4, 5]]).fit()
model.rsquared

model.summary()


lr_model = ols(formula='st_mass~st_rad', data=new_df).fit()
lr_model.summary()
sns.jointplot(new_df['st_optmag'], new_df['st_teff'], kind='reg')

df.columns
del new_df["st_metratio"]
new_df.isna().sum()
new_df.shape
new_df.rename(columns={'pl_name': 'Planet_Name',
                       "sy_snum": "num_of_stars",
                       "sy_pnum": "num_of_planets",
                       "st_teff": "st_temperature",
                       "st_rad": "st_radius",
                       "st_met": "st_metalicity",
                       "st_lum": "log_st_lum",
                       "st_logg": "log_st_gravity"}, inplace=True)
new_df.head(2)

# log transformation

cols = ["st_temperature", "st_radius", "st_mass"]

for col in cols:
    col_log = "log_"+col
    new_df[col_log] = np.log(new_df[col])


################################################################################

#


star_df = df[np.isfinite(df["st_mass"])]
star_df = star_df[np.isfinite(star_df["st_rad"])]
star_df = star_df[np.isfinite(star_df["st_teff"])]
star_df.isna().sum()
star_df.shape
lr_model = ols(formula='st_mass~st_rad+st_teff', data=star_df).fit()
lr_model.summary()

new_df = star_df[['st_mass', 'st_rad', 'st_teff']]
cols = list(new_df.columns)
cols


# making log columns
def columns_log(coln, data):
    for col in cols:
        col_log = "log_"+col
        data[col_log] = np.log(data[col])


new_df.head(2)


lr_model_rad = ols(formula='log_st_mass~log_st_rad', data=new_df).fit()
lr_model_rad.summary()
lr_model_teff = ols(formula='log_st_mass~log_st_teff', data=star_df).fit()
lr_model_teff.summary()


def qqplot(x, y, data):
    lr_model = ols(formula=f"{y}~{x}", data=data).fit()
    pred_val = lr_model.fittedvalues.copy()
    true_val = data[f'{y}'].values.copy()
    residual = true_val - pred_val
    fig, ax = plt.subplots(figsize=(6, 2.5))
    sp.stats.probplot(residual, plot=ax, fit=True)


hf.qqplot('log_st_teff', 'log_st_mass', new_df)


qqplot('log_st_teff', 'log_st_mass', star_df)
qqplot('log_st_rad', 'log_st_mass', star_df)


def scater(x, y, data):
    sns.jointplot(data[f'{x}'], data[f'{y}'], kind='reg')


scater('log_st_mass', 'log_st_rad', star_df)

scater('log_st_mass', 'log_st_teff', star_df)
