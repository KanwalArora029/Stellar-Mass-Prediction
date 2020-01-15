import scipy as sp
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')


# read csv
df = pd.read_csv("/Users/flatironschool/Documents/flatiron/Mod4_project/Data/star_table1.csv",
                 error_bad_lines=False)
df.head(3)
