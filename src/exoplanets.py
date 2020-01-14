import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')


# read csv
df = pd.read_csv("/Users/flatironschool/Documents/flatiron/Mod4_project/Data/planets_table.csv")
df.head(3)
df.columns
df = df[["pl_name", "pl_discmethod", "pl_pnum", "pl_orbper", "pl_orbeccen",
         "pl_bmassj", "pl_radj"]]
