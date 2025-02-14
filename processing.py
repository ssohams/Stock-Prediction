import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

from joblib import dump, load

import matplotlib.pyplot as plt

stock_data = pd.read_csv('all_stocks_5yr.csv')

#creating daily_return and volatility features
stock_data["daily_return"] = stock_data['open']-stock_data['close']
stock_data['rolling_vol'] = stock_data['daily_return'].rolling(window=30).std()
stock_data['moving_avg'] = stock_data['close'].rolling(window=7).mean()

#defining extra features
X = stock_data[['open','high','low','close','daily_return','rolling_vol','moving_avg']]
y = stock_data['close'].shift(-1)

print(type(X))
X.drop(index=X.index[-1],axis=0,inplace=True)
print(type(y))
y.drop(index=y.index[-1],axis=0,inplace=True)
print(X)
print(y)

#training 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=False)
print("Completed split...")

random_forest = RandomForestRegressor(n_estimators=100,random_state=10)
random_forest.fit(X_train,y_train)
print('Fit complete...')


dump(random_forest, 'random_forest_model.joblib')
X_test.to_csv('X_test.csv')
y_test.to_csv('y_test.csv')

print("Complete.")
'''
loaded_forest = load('random_forest_model.joblib')

#Model evaluation
y_pred = loaded_forest.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
mape = (abs((y_pred - y_test) / y_test).mean())* 100 



print(f'Standard Deviation: {mse}')
print(f'Mean Absolute Percent Error: {mape}')



plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.show()
'''






