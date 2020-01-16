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
df.columns
planet_df = df[["pl_name", "pl_discmethod", "pl_pnum", "pl_orbper", "pl_orbeccen",
         "pl_bmassj", "pl_radj"]]
planet_df.head()

planet_df.isna().sum()


#star planets_table
star_df = df[["pl_pnum","st_optmag", "gaia_gmag", "st_teff", "st_mass", "st_rad"]]
star_df.head(3)

df = df[["pl_name", "pl_discmethod", "pl_pnum", "pl_orbper", "pl_orbeccen",
         "pl_bmassj", "pl_radj","st_optmag", "gaia_gmag", "st_teff", "st_mass", "st_rad"]]
df
plt.scatter(df.pl_radj, df.st_mass);
sns.boxplot(x=df["st_mass"])
from scipy import stats
z = np.abs(stats.zscore(df.pl_orbper))
df.dtypes
cols = list(df.columns)
cols.remove("pl_name")
cols.remove("pl_discmethod")
cols
for col in cols:
    col_zscore = col + '_zscore'
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
df.head(2)

df.columns
zscore_df = df[['pl_name','pl_pnum_zscore', 'pl_orbper_zscore', 'pl_orbeccen_zscore',
'pl_bmassj_zscore', 'pl_radj_zscore', 'st_optmag_zscore',
'gaia_gmag_zscore', 'st_teff_zscore', 'st_mass_zscore',
'st_rad_zscore']]
df = df[['pl_name', 'pl_discmethod', 'pl_pnum', 'pl_orbper', 'pl_orbeccen',
       'pl_bmassj', 'pl_radj', 'st_optmag', 'gaia_gmag', 'st_teff', 'st_mass',
       'st_rad']]



for col in cols:
    col_zscore = col + '_zscore'
    zscore_df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
df

plt.scatter(df["log(pl_radj)"], df["log(st_mass)"]);

sns.boxplot(x=df["log(st_mass)"])
plt.hist(df["log(st_mass)"], bins = 1000);
df.columns
df = df[['pl_name', 'pl_discmethod', 'pl_pnum', 'pl_orbper', 'pl_orbeccen',
       'pl_bmassj', 'pl_radj', 'st_optmag', 'gaia_gmag', 'st_teff', 'st_mass',
       'st_rad']]
for col in cols:
    col_zscore = "log_"+col
    df[col_zscore] = np.log(df[col])
df.head(2)


lr_model = ols(formula='log_st_mass~log_pl_radj', data=df).fit()
lr_model.summary()



pred_val = lr_model.fittedvalues.copy()
true_val = df['log_st_mass'].values.copy()
residual = true_val - pred_val
df.mean()
star_df.isna().sum()
star_df_clean = star_df[star_df.isna().sum(axis=1) <=1]
star_df_clean.isna().sum()
star_nonan_df = star_df.dropna()
star_nonan_df.shape
star_nonan_df.head()
star_nonan_df.columns

cols = list(star_nonan_df.columns)
cols.remove("pl_pnum")
cols
for col in cols:
    col_zscore = "log_"+col
    star_nonan_df[col_zscore] = np.log(df[col])


lr_model = ols(formula='pl_pnum~log_st_optmag+log_gaia_gmag+log_st_teff+log_st_mass+log_st_rad', data=df).fit()
lr_model.summary()
