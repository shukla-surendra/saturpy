"""
this is driver code for initial model testing
"""
from saturpy.regressor import LinearRegression
from saturpy.datasets import load_iris

# data
df = load_iris()
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
X_train, y_train, X_test, y_test = X[0:100], y[0:100], X[150:], y[100:]

# initialize Model
linear_regression = LinearRegression()

# training the model
linear_regression.fit(X_train, y_train)

# Predicting the model
prediction = linear_regression.predict(X_test)
print("Prediction", prediction)
