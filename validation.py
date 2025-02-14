import pandas as pd

from joblib import dump, load

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


import numpy as np


loaded_forest = load('random_forest_model.joblib')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

#fix the shape of the dfs
X_test = X_test.drop(columns=['Unnamed: 0'])
y_test = y_test.drop(columns =['Unnamed: 0'])

y_test = y_test.iloc[:, 0]



#Model evaluation
y_pred = loaded_forest.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
mape = (abs((y_pred - y_test) / y_test).mean())* 100 




print(f'Standard Deviation: {mse}')
print(f'Mean Absolute Percent Error: {mape}')




'''
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.show()
'''
open_price = 100.0
high_price = 105.0
low_price = 98.0
close_price = 102.0
daily_return = open_price - close_price
rolling_vol = 2.5  # Example, compute this based on past 30 days
moving_avg = 101.5  # Example, compute this based on past 7 days

# Create a NumPy array in the correct shape (1 sample, 7 features)
input_data = np.array([[open_price, high_price, low_price, close_price, daily_return, rolling_vol, moving_avg]])

predicted_close = loaded_forest.predict(input_data)

print(f'Predicted Close Price: {predicted_close[0]}')





