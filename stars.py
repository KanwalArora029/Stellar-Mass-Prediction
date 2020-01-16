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
df = pd.read_csv("/Users/flatironschool/Documents/flatiron/Mod4_project/Data/planets_stars.csv",
                 error_bad_lines=False)
df.isna().sum()

new_df = df[np.isfinite(df["pl_trandur"])]
new_df = new_df[np.isfinite(new_df["pl_orbsmax"])]
new_df.shape
new_df = new_df[np.isfinite(new_df["st_mass"])]
new_df = new_df[np.isfinite(new_df["st_rad"])]
new_df = new_df[np.isfinite(new_df["st_teff"])]
new_df.shape
new_df.isna().sum()


lr_model = ols(formula='st_mass', data=new_df).fit()
lr_model.summary()


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
