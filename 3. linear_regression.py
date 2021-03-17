# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 00:04:35 2021

@author: leeso
"""

# LinearRegression (using statsmodels)
fit_reg1 = sm.OLS(Y_train, X_train).fit()
display(fit_reg1.summary())
pred_tr_reg1 = fit_reg1.predict(X_train).values
pred_te_reg1 = fit_reg1.predict(X_test).values