import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Boston = load_data("Boston")

# .shape[0] returns numbers of row // np.ones(X) returns an array of X size containing full of 1.(s)
X = pd.DataFrame(data={'intercept': np.ones(Boston.shape[0]), 'lstat': Boston['lstat']}) 

y = Boston['medv']
model = sm.OLS(y, X)    # Specify Ordinary Least Square model
results = model.fit()   # fit/train the model
#print(results.summary()) # Info about the model

new_df = pd.DataFrame({'lstat':[5, 10, 15]})
design = MS(['lstat']).fit(Boston);
newX = design.transform(new_df);

new_predictions = results.get_prediction(newX);
print(new_predictions.predicted_mean)
print(new_predictions.conf_int(alpha=0.05))
print(new_predictions.conf_int(obs=True, alpha=0.05))

'----------------------------------------------------'
def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)
'----------------------------------------------------'

ax = Boston.plot.scatter('lstat', 'medv')
abline(ax, results.params[0], results.params[1], 'r--', linewidth=3)


infl = results.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)


fig, ax = subplots(figsize=(8,8))
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');
fig.show()


"""
==============================================================================================
"""


