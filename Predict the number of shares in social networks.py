import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import requests
from io import StringIO

data_url = 'https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Online%20News%20Popularity/OnlineNewsPopularity.csv'
response = requests.get(data_url)

data = pd.read_csv(StringIO(response.text))

print(data.columns)

target_variable =' shares'

if target_variable not in data.columns:
    raise KeyError(f"The target variable '{target_variable}' is not found in the dataset.")

features = data.drop(columns=['url', target_variable])

target = data[target_variable]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.4f}")

linear_regression = LinearRegression()
decision_tree_regression = DecisionTreeRegressor(random_state=42)
random_forest_regression = RandomForestRegressor(random_state=42)
gradient_boosting_regression = GradientBoostingRegressor(random_state=42)

train_and_evaluate_model(linear_regression, 'Linear Regression')
train_and_evaluate_model(decision_tree_regression, 'Decision Tree Regression')
train_and_evaluate_model(random_forest_regression, 'Random Forest Regression')
train_and_evaluate_model(gradient_boosting_regression, 'Gradient Boosting Regression')

plt.scatter(y_test, gradient_boosting_regression.predict(X_test))
plt.xlabel("Actual Shares")
plt.ylabel("Predicted Shares (Gradient Boosting)")
plt.title("Actual Shares vs. Predicted Shares")
plt.show()


After Tuning :


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import requests
from io import StringIO

data_url = 'https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Online%20News%20Popularity/OnlineNewsPopularity.csv'
response = requests.get(data_url)

data = pd.read_csv(StringIO(response.text))

target_variable = ' shares'

if target_variable not in data.columns:
    raise KeyError(f"The target variable '{target_variable}' is not found in the dataset.")

features = data.drop(columns=['url', target_variable])

target = data[target_variable]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

linear_regression = LinearRegression()
decision_tree_regression = DecisionTreeRegressor(random_state=42)

linear_regression_param_grid = {}  

decision_tree_param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

linear_regression_grid_search = GridSearchCV(estimator=linear_regression, param_grid=linear_regression_param_grid, cv=3, n_jobs=-1, verbose=2)
decision_tree_grid_search = GridSearchCV(estimator=decision_tree_regression, param_grid=decision_tree_param_grid, cv=3, n_jobs=-1, verbose=2)

linear_regression_grid_search.fit(X_train, y_train)
decision_tree_grid_search.fit(X_train, y_train)

linear_regression_best_params = linear_regression_grid_search.best_params_
linear_regression_best_model = linear_regression_grid_search.best_estimator_

decision_tree_best_params = decision_tree_grid_search.best_params_
decision_tree_best_model = decision_tree_grid_search.best_estimator_

linear_regression_y_pred = linear_regression_best_model.predict(X_test)
linear_regression_mse = mean_squared_error(y_test, linear_regression_y_pred)
linear_regression_r2 = r2_score(y_test, linear_regression_y_pred)

decision_tree_y_pred = decision_tree_best_model.predict(X_test)
decision_tree_mse = mean_squared_error(y_test, decision_tree_y_pred)
decision_tree_r2 = r2_score(y_test, decision_tree_y_pred)

print("Linear Regression - Best Hyperparameters:", linear_regression_best_params)
print("Linear Regression - Mean Squared Error:", linear_regression_mse)
print("Linear Regression - R-squared:", linear_regression_r2)

print("Decision Tree Regression - Best Hyperparameters:", decision_tree_best_params)
print("Decision Tree Regression - Mean Squared Error:", decision_tree_mse)
print("Decision Tree Regression - R-squared:", decision_tree_r2)
