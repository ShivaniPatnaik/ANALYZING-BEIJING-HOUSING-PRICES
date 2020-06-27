from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('factors.csv', encoding ='iso-8859-1')
data.head()
X = data.drop(columns=['house_prices'])
Y = data['house_prices']

#Forecast
forecast_out = 5
data['Prediction'] = data[['house_prices']].shift(-forecast_out)
print(data.tail())
X = np.array(data.drop(['Prediction'],1))
X = X[:-forecast_out]
print(X)
y = np.array(data['Prediction'])
y = y[:-forecast_out]
print(y)
# Set x_forecast equal to the last 5 rows of the original data set from house_prices column
x_forecast = np.array(data.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

#Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
# with statsmodels
X = sm.add_constant(X) # adding a constant
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)  
print_model = model.summary()
print(print_model)

#Create and train the Linear Regression  Model
lr = LinearRegression()
lr.fit(x_train, y_train)
r_sq = lr.score(X, y)
print('coefficient of determination:', r_sq)
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

#correlation
correlation = data.corr(method='pearson')
print(correlation)  
ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(data.corr(), annot = True, cmap ="PiYG", linewidths = 0.1)