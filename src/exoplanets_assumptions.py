import scipy as sp
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')


# read csv
df = pd.read_csv("/Users/flatironschool/Documents/flatiron/Mod4_project/Data/planets_table.csv")
df.head(3)


df = df[["pl_name", "pl_discmethod", "pl_pnum", "pl_orbper", "pl_orbeccen",
         "pl_bmassj", "pl_radj", "st_optmag", "gaia_gmag", "st_teff", "st_mass", "st_rad"]]

sns.scatterplot(df.pl_radj, df.st_mass)

sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
# sns.despine(left=True)
sns.jointplot("st_optmag", "pl_bmassj", data=df)
sns.scatterplot(df.st_optmag, df.pl_bmassj, color="b", ax=axes[0, 0])
sns.scatterplot(df.pl_radj, df.st_mass, color="b", ax=axes[0, 0])
sns.scatterplot(df.pl_radj, df.st_mass, color="b", ax=axes[0, 0])
sns.scatterplot(df.pl_radj, df.st_mass, color="b", ax=axes[0, 0])

# read csv
new_df = pd.read_csv(
    "/Users/flatironschool/Documents/flatiron/Mod4_project/Data/exoplanet_table.csv")
new_df.head(3)
new_df.shape
new_df = new_df[new_df.st_teff.isna()]
new_df.isna().sum()

new_df = new_df[np.isfinite(new_df["st_mass"])]
new_df = new_df[np.isfinite(new_df["st_rad"])]
new_df.isna().sum()
new_df.shape
lr_model = ols(formula='st_mass~st_rad+st_teff', data=new_df).fit()
lr_model.summary()

star_df = new_df[['sy_snum', 'sy_pnum', 'st_mass', 'st_rad', 'st_teff']]
cols = list(star_df.columns)
cols
cols.remove("sy_snum")
cols.remove("sy_pnum")

# making log columns
for col in cols:
    col_log = "log_"+col
    star_df[col_log] = np.log(df[col])

star_df.head(2)


lr_model = ols(formula='log_st_mass~log_st_rad', data=star_df).fit()
lr_model.summary()

sns.jointplot(star_df['log_st_mass'], star_df['log_st_rad'], kind='reg')

sns.jointplot(star_df['log_st_mass'], star_df['log_st_teff'], kind='reg')

# checking for our model - Homoscedasticity
pred_val = lr_model.fittedvalues.copy()
true_val = star_df['log_st_mass'].values.copy()
residual = true_val - pred_val

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.scatter(star_df['log_st_rad'], residual)


fig, ax = plt.subplots(figsize=(6, 2.5))
sp.stats.probplot(residual, plot=ax, fit=True)
