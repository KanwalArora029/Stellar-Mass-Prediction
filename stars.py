import helpf as hf
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
new_df.head(10)
X = new_df[["pl_trandur", "pl_orbsmax"]]

Y = new_df['st_mass'].apply(lambda x: x*1.989e+30)
X['pl_trandur_metric'] = X['pl_trandur'].apply(lambda x: x*8.64e+4)
X['pl_orbsmax_metric'] = X['pl_orbsmax'].apply(lambda x: x*1.496e+8)

X_rad = new_df[["pl_trandur", "pl_orbsmax", "pl_radj"]]

ln_df = new_df[["st_mass", "pl_trandur", "pl_orbsmax"]]

X.columns
cols = ['pl_trandur_metric', 'pl_orbsmax_metric']


for col in cols:
    col_ln = "log_"+col
    X[col_ln] = np.log(X[col])

X_metric = X[['log_pl_trandur_metric', 'log_pl_orbsmax_metric']]
Y_metric = np.log(Y)

X_metric = sm.add_constant(X_metric)
X_metric["solar_mass"] = Y_metric


model_metric = ols(
    formula="solar_mass~log_pl_trandur_metric+log_pl_orbsmax_metric", data=X_metric).fit()
model_metric.summary()

X_metric.drop(["solar_mass"], axis=1, inplace=True)

np.log((np.pi**2*4)/6.67e-11)

model_metric = sm.OLS(Y_metric, X_metric).fit()
model_metric.params
model_metric.summary()

ln_df["pl_orbsmax3"] = ln_df["pl_orbsmax"]**3

ln_df["pl_trandur2"] = ln_df["pl_trandur"]**2


ln_df.head(2)


ln_model = ols(formula='log_st_mass~log_pl_orbsmax+log_pl_trandur', data=ln_df).fit()
ln_model.summary()


X_zcore = pd.DataFrame()

for col in cols:
    col_zscore = col + '_zscore'
    X_zcore[col_zscore] = (X[col] - X[col].mean())/X[col].std(ddof=0)
X_rad_zcore = pd.DataFrame()

cols = X_rad.columns

for col in cols:
    col_zscore = col + '_zscore'
    X_rad_zcore[col_zscore] = (X_rad[col] - X_rad[col].mean())/X_rad[col].std(ddof=0)

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

new_df.head(2)
new_df.isna().sum()
new_df = new_df[np.isfinite(new_df["log_st_gravity"])]
new_df.shape

lr_model_rad = ols(formula='log_st_mass~log_st_radius', data=new_df).fit()
lr_model_rad.summary()
lr_model_teff = ols(formula='log_st_mass~log_st_temperature', data=new_df).fit()
lr_model_teff.summary()
sns.jointplot(new_df['log_st_mass'], new_df['log_st_radius'], kind='reg')

new_df.columns

sns.jointplot(new_df['log_st_mass'], new_df['st_metalicity'], kind='reg')


new_df.shape


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


new_df = new_df[np.isfinite(new_df["st_metalicity"])]
regression(new_df['st_metalicity'], new_df['st_mass'], "red")

# taking care of missing Data
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(new_df.iloc[:, 5:-1])
new_df.iloc[:, 5:-1] = imputer.transform(new_df.iloc[:, 5:-1])
new_df.isna().sum()
new_df
new_df.num_of_planets.value_counts()


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
for col in cols:
    col_log = "log_"+col
    new_df[col_log] = np.log(df[col])

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
