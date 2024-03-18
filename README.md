# Linear_Regression25
Working with Linear Regression**
``` python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# fetch_openml is a function that fetches the in-built sklearn datasets. 
from sklearn.datasets import fetch_openml

boston = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
data = boston.data # This variable contains all the independent variables of the DataFrame
target = boston.target # This variable contains the dependent variable
# This variables contains all the features present in the dataset. Selecting the right features will be important when building the model.
feature_names = boston.feature_names


df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['HousePrice'] = boston.target
df.head()

# Observation: This is the head of the dataset. It shows the top five rows of the DataFrame.

df.describe()

# Observation: Here, you can see the statistical analysis of the dataset.

df.isna().sum()
# Observation: There is no missing data in the dataset. It does not require any null value treatment.
sns.boxplot(df['HousePrice'])

#Observation: There are outliers in the dataset.

# Setting the upper limit for the HousePrice column to the 99% quantile value.
upper_limit = df['HousePrice'].quantile(0.99)
# Setting the lower limit for the HousePrice column to the 1% quantile value. 
lower_limit = df['HousePrice'].quantile(0.01)

# Ensure that HousePrice values below the lower limit are adjusted to the lower limit
df['HousePrice'] = np.where(df['HousePrice'] < lower_limit, lower_limit, df['HousePrice'])
# Ensure that HousePrice values above the upper limit are adjusted to the upper limit
df['HousePrice'] = np.where(df['HousePrice'] > upper_limit, upper_limit, df['HousePrice'])

import statsmodels.api as sm
import numpy as np
import pandas as pd

boston.data = boston.data.apply(pd.to_numeric)

X_constant = sm.add_constant(np.asarray(boston.data))
# The ordinary least squares (OLS) algorithm is a method for estimating the parameters of a linear regression model.
boston_model = sm.OLS(boston.target, np.asarray(boston.data)).fit()
boston_model.summary()

def calculate_residuals(model, features, label):
# Predict the labels using the provided model and input features
    predictions =  model.predict(features)
# Calculate the residuals by subtracting predicted values from actual values
    df_results = pd.DataFrame({'Actual' : label, 'Predicted' : predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    return df_results

def linear_assumptions(model, features, label):
    df_results = calculate_residuals(model, features, label)

    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, height=7)
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords, color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()

linear_assumptions(boston_model, boston.data, boston.target)

# Observation: We can observe that the line does not represent all the data points.

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

x = df.drop(['HousePrice'], axis=1)
x = x.astype(float)  # Convert the array to float type

vif_data = pd.DataFrame()
vif_data['Features'] = x.columns

vif_data['vif'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)

# Observation: From the above output, we can infer that the columns NOX, RM, AGE, and PTRATIO have higher multicollinearity. Hence, we can drop them.

df1 = df.drop(['NOX', 'RM', 'AGE', 'PTRATIO'], axis = 1)

x = df1.drop(['HousePrice'], axis =1)
y = df1['HousePrice']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(x, y, random_state=0, test_size=0.25)
# Here the train and test datasets are using the 75:25 split ratio.

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

X_train = X_train.apply(pd.to_numeric, errors='coerce')

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

print(X_train.dtype)
print(y_train.dtype)
print(np.isnan(X_train).sum())
print(np.isnan(y_train).sum())

model = sm.OLS(y_train, X_train).fit()

print(model.summary())

# Import the necessary libraries to evalutate the model performance.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_pred_train = reg.predict(X_train)
#y_pred_test = reg.predict(X_test)

X_test = pd.DataFrame(X_test)  # Convert X_test to a pandas DataFrame
X_test = X_test.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
X_test = np.asarray(X_test)  # Convert X_test to a numpy array

y_pred_test = reg.predict(X_test)  # Predict using the trained model

print("R Square: {}".format(r2_score(y_train, y_pred_train)))
print("MAE: {}".format(mean_absolute_error(y_train, y_pred_train)))
print("MSE: {}".format(mean_squared_error(y_train, y_pred_train)))
#Observation: From the above output, we can observe that the model is a moderate fit for the given training dataset.

print("R Square: {}".format(r2_score(y_test, y_pred_test)))
print("MAE: {}".format(mean_absolute_error(y_test, y_pred_test)))
print("MSE: {}".format(mean_squared_error(y_test, y_pred_test)))

#Observation: The model moderately explains the testing data, as indicated by the R Square value.

```
Copy downloadable file for WS, Demo answers above 
```
https://vocproxy-1-21.us-west-2.vocareum.com/files/home/labsuser/Block_25_Demo_1_Student.ipynb?_xsrf=2%7C3db90129%7Cacba5a54690658c9a4998717e4af4654%7C1708812522
```
