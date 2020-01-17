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

# open helpf.py from a differen directory
sys.path.insert(0, '/Users/flatironschool/Documents/flatiron/Mod4_project/Notebooks')

# read csv
df = pd.read_csv(
    "/Users/flatironschool/Documents/flatiron/Mod4_project/Data/planets_stars.csv",
    error_bad_lines=False)
df.isna().sum()

# getting rid of NaN's in stellar columns, pl_orbsmax and pl_trandur
new_df = df[np.isfinite(df["pl_trandur"])]
new_df = new_df[np.isfinite(new_df["pl_orbsmax"])]
new_df = new_df[np.isfinite(new_df["st_mass"])]
new_df = new_df[np.isfinite(new_df["st_rad"])]
new_df = new_df[np.isfinite(new_df["st_teff"])]
new_df.shape
# after data selection there are only 514 rows
new_df.isna().sum()

# data frame with only planet distance from star "pl_orbsmax" and planets
# transit duration
X = new_df[["pl_trandur", "pl_orbsmax"]]

# changing units to metric system.
Y = new_df['st_mass'].apply(lambda x: x*1.989e+30)
X['pl_trandur_metric'] = X['pl_trandur'].apply(lambda x: x*8.64e+4)
X['pl_orbsmax_metric'] = X['pl_orbsmax'].apply(lambda x: x*1.496e+8)

X.columns
cols = ['pl_trandur_metric', 'pl_orbsmax_metric']

# adding log transformatin columns to the dataframe.
hf.columns_log(cols, X)
X.columns


X_log = X[['log_pl_trandur_metric', 'log_pl_orbsmax_metric']]
Y_log = np.log(Y)
Y
X_cons = sm.add_constant(X_log)

model_metric = sm.OLS(Y_log, X_log).fit()
model_metric.summary()
model_metric.params

model_metric = sm.OLS(Y_log, X_cons).fit()
model_metric.summary()


hf.columns_zc(cols, X)
Y_zc = (Y - Y.mean())/Y.std(ddof=0)

X.head(2)

################################################################################

# trying out centered model

model = sm.OLS(Y_zc, X.iloc[:, [6, 7]]).fit()
model.rsquared
