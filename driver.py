"""
this is driver code for initial model testing
"""
from saturpy.regressor import LinearRegression

# data
X_train, y_train, X_test, y_test = [], [], [], []

# initialize Model
linear_regression = LinearRegression()

# training the model
linear_regression.fit(X_train, y_train)

# Predicting the model
prediction = linear_regression.predict(X_test)
print(prediction)
