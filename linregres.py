# Linear regression model for heart data using the OLS test. This model compares age and maximum heart rate 
# to see if there is a correlation between them. Through this model it shows that there is not a very strong 
# correlation as the r squared value is around 0.2, which shows that it only accounts for 20% of the variation 
# in the data. 
# Jamacyn Davis 
# June 28, 2021 

# Supress warnings 
import warnings 
warnings.filterwarnings('ignore')

# Import numpy 
import numpy as np 
import pandas as pd 

# Allowing the user to choose the file
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)

# Read in the given CSV file 
heart = pd.read_csv(filename)
# heart = cudf.read_csv('/data/sample.csv')
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
sns.pairplot(heart, x_vars=['age', 'trtbps'], y_vars='thalachh', size = 5, aspect = 1, kind = 'scatter')
plt.show()

# Creating a heatmap represenation of the correlation
figure = plt.figure(figsize = (10, 10))
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
fig = plt.figure(figsize = (10, 10)) 
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




#Attempting to add GUI controls 
import tkinter as tk 
from tkinter import * 
from tkinter.constants import BOTH
from PIL import Image, ImageTk

class Window(Frame):
    # Creates basic window 
    def __init__(self, master=None):        
        Frame.__init__(self, master)
        self.master = master 
        self.init__window() 

    def init__window(self): 
        self.master.title("Regression Model")
        self.pack(fill = BOTH, expand = 1) 

        # quitButton = Button(self, text = "Quit", command = self.client_exit)
        # quitButton.place(x=0, y=0)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        file = Menu(menu)
        file.add_command(label='Exit', command=self.client_exit)
        menu.add_cascade(label='File', menu=file)
        edit = Menu(menu)
        # edit.add_command(label='Add Image', command=self.showImg)
        # edit.add_command(label='Add Text', command=self.showTxt)
        menu.add_cascade(label='Edit', menu=edit)

        # Scatter plot button and label
        SPButton = Button(self, text = 'Scatter Plot', command = self.showImg)
        SPButton.place(x = 350, y = 150)
        SPLabel = Label(self, text = 'Display Scatter Plot')
        SPLabel.place(x = 10, y = 150)

        # Heat map button and label 
        HMButton = Button(self, text = 'Heat Map')
        HMButton.place(x = 350, y = 200)
        HMLabel = Label(self, text = 'Display Heat Map')
        HMLabel.place(x = 10, y = 200)

        # Regression line button and label
        RLButton = Button(self, text = 'Regression Line')
        RLButton.place(x = 350, y = 250)
        RLLabel = Label(self, text = 'Display Regression Line on Sactter PLot')
        RLLabel.place(x = 10, y = 250)

        # Scatter plot with residual 
        RSPButton = Button(self, text = 'Residual')
        RSPButton.place(x = 350, y = 300)
        RSPLabel = Label(self, text = 'Display the residual on a scatter plot')
        RSPLabel.place(x = 10, y = 300)

    def client_exit(self): 
        exit()  

    def showImg(self):
        load = Image.open('ScatterPlots.png')
        render = ImageTk.PhotoImage(load) 
        img = Label(self, image=render)
        img.image = render 
        img.place(x=0, y=0)

root = Tk() 
root.geometry('600x600')
app = Window(root)
root.mainloop()

