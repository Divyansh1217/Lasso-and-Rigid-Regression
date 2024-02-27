import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
data=pd.read_csv("fuel_EFF.csv")
print(data.head())
data.drop(["Unnamed: 0"],axis=1,inplace=True)
def scaplot(feature,target):
    plt.figure(figsize=(16,18))
    plt.scatter(data[feature],data[target],c="black")
    plt.xlabel("Money Spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    st.pyplot(plt.show())
scaplot("TV", "Sales")
scaplot("Radio", "Sales")
scaplot("Newspaper", "Sales")
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

xs = data.drop(["Sales"], axis=1)
y = data["Sales"].values.reshape(-1,1)
linreg = LinearRegression()
MSE = cross_val_score(linreg, xs, y, scoring="neg_mean_squared_error", cv=5)

mean_MSE = np.mean(MSE)
st.write(mean_MSE)


# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()

parameters = {"alpha":[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regression = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regression.fit(xs, y)
st.write(ridge_regression.best_params_)
st.write(ridge_regression.best_score_)


from sklearn.linear_model import Lasso
lasso = Lasso()

parameters = {"alpha":[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regression = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regression.fit(xs, y)

st.write(lasso_regression.best_params_)
st.write(lasso_regression.best_score_)