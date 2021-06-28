# Linear regression model for heart data using the OLS test. 
# Jamacyn Davis 
# June 28, 2021 

# Supress warnings 
import warnings 
warnings.filterwarnings('ignore')


# Import numpy 
import numpy as np 
import pandas as pd 

# Read in the given CSV file 
heart = pd.read_csv('heart.csv')
print(heart)

# Indicates the number of dimensions
heart.shape

# Used for statistical representations of the data 
heart.describe()

# Prints information like the total number of data entries and null values
heart.info()

# Import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Creating a pairplot of the data using seaborn
sns.pairplot(heart, x_vars=['age', 'trtbps', 'sex'], y_vars='thalachh', size = 6, aspect = 1, kind = 'scatter')
plt.show()


# Creating a heatmap represenation of the correlation
sns.heatmap(heart.corr(), cmap = 'PuBu', annot=True)
plt.xlabel('Values on x axis')
plt.ylabel('Values on y axis')
plt.show() 


x = heart['age']
y = heart['thalachh']

# Splitting the data into train and test sets (100% random)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state = 100)

print("\n")
print(x_train)
print(y_train)

import statsmodels.api as model 
x_train_sm = model.add_constant(x_train)

lr = model.OLS(y_train, x_train_sm).fit()
print(lr.params)
print(lr.summary())

plt.scatter(x_train, y_train)
plt.plot(x_train, 201.157201 - 0.949434 * x_train, 'r')
plt.show()

#predicting the y values based on the regression line 
y_train_pred = lr.predict(x_train_sm)

#residual 
res = (y_train - y_train_pred)

# Creating a histogram using the residual values found earlier
fig = plt.figure() 
sns.distplot(res, bins=10)
plt.title("Error", fontsize = 11)
plt.xlabel('y_train - y_train_pred', fontsize = 10)
plt.show()

# making sure no patterns were missed 
plt.scatter(x_train, res)
plt.show() 

# Calculating r squared 
from sklearn.metrics import r2_score 
r_2_value = r2_score(y_train, y_train_pred)
print('r-squared value: ')
print(r_2_value)
# Now dealing with the test data 

# adding a constant to the test data for x 
x_test_sm = model.add_constant(x_test)

# predict 
y_test_pred = lr.predict(x_test_sm)
print(y_test_pred)

r_squared  = r2_score(y_test, y_test_pred)
print('r squared value: ')
print(r_squared)

plt.scatter(x_test, y_test)
plt.plot(x_test, y_test_pred, 'r')
plt.show()

# Comparing the r squared value from the train and test data 
print('\nThis is the r squared value from the test data: ')
print(r_squared)
print('This is the r squared data from the train data: ')
print(r_2_value)