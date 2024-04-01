# This is a our utility module or a helper module, which contanins all the functions that are used throughout our project

# Libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt 

def model_training(model_name, X, y, params=None):
    # Select model based on user choice
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Ridge':
        model = Ridge()
    elif model_name == 'Lasso':
        model = Lasso()
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators= params)
    elif model_name == 'SVM':
        model = SVR(kernel = params)
    elif model_name == 'XGBoost':
        model = XGBRegressor(objective ='reg:absoluteerror', booster = params[0], n_estimators = params[1])
    elif model_name == 'KNN':
        model = KNeighborsRegressor(n_neighbors = params)
    else:
        return "Please check your model name!"

    # Train the model
    model.fit(X, y)

    return model

def test_model(model,X,y):
    # Make predictions on the test set
    y_pred = model.predict(X)

    # Evaluate the model
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return mae, r2


def plot_model(model,X,y,title="Actual vs Predicted Values"):
    # Make predictions on the test set
    y_pred = model.predict(X)

    # Plot the predicted values against the actual values
    plt.scatter(X.index, y, label='Actual')
    plt.scatter(X.index, y_pred, label='Predicted')
    plt.legend()
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.show()

