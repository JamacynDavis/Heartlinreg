# Linear regression model for heart data using the OLS test. This model compares age and maximum heart rate 
# to see if there is a correlation between them. Through this model it shows that there is not a very strong 
# correlation as the r squared value is around 0.2, which shows that it only accounts for 20% of the variation 
# in the data which is provided by Kaggle.com. 
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
from tkinter.filedialog import askopenfilename, asksaveasfile

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
# sns.pairplot(heart, x_vars=['age', 'trtbps'], y_vars='thalachh', size = 5, aspect = 1, kind = 'scatter')
# plt.show()

# Creating a heatmap represenation of the correlation
# figure = plt.figure(figsize = (10, 10))
# sns.heatmap(heart.corr(), cmap = 'PuBu', annot=True)
# plt.xlabel('Values on x axis')
# plt.ylabel('Values on y axis')
# plt.show() 

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

# plt.scatter(x_train, y_train)
# plt.plot(x_train, 201.157201 - 0.949434 * x_train, 'r')
# plt.show()

#predicting the y values based on the regression line 
y_train_pred = lr.predict(x_train_sm)

#residual 
res = (y_train - y_train_pred)

# Creating a histogram using the residual values found earlier
# fig = plt.figure(figsize = (6, 6)) 
# sns.distplot(res, bins=10)
# plt.title("Error", fontsize = 11)
# plt.xlabel('y_train - y_train_pred', fontsize = 10)
# plt.show()

# making sure no patterns were missed 
# plt.scatter(x_train, res)
# plt.show() 

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

# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_test_pred, 'r')
# plt.show()

# Comparing the r squared value from the train and test data 
print('\nThis is the r squared value from the test data: ')
print(r_squared)
print('This is the r squared data from the train data: ')
print(r_2_value)


#Attempting to add GUI controls 
import tkinter as tk 
from tkinter import * 
from tkinter.constants import BOTH
from PIL import Image

class Window(Frame):
    # Creates basic window 
    def __init__(self, master=None):        
        Frame.__init__(self, master)
        self.master = master 
        self.init__window() 

    def init__window(self): 
        self.master.title("Regression Model")
        self.pack(fill = BOTH, expand = 1) 

        quitButton = Button(self, text = "x", command = self.client_exit, bg = 'Red')
        quitButton.place(x=1325, y=0)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        file = Menu(menu)
        file.add_command(label='Exit', command=self.client_exit)
        menu.add_cascade(label='File', menu=file)
        edit = Menu(menu)
        menu.add_cascade(label='Edit', menu=edit)


        # Heat map button and label 
        HMButton = Button(self, text = 'Heat Map', command = self.showImg2, bg='MediumPurple1')
        HMButton.place(x = 500, y = 200)
        HMLabel = Label(self, text = 'Display heat map', font = 'Latha')
        HMLabel.place(x = 10, y = 200)

        # Regression line button and label
        RLButton = Button(self, text = 'Regression Line', command = self.showImg3, bg='MediumPurple1')
        RLButton.place(x = 500, y = 250)
        RLLabel = Label(self, text = 'Display regression line on scatter plot', font = 'Latha')
        RLLabel.place(x = 10, y = 250)

        # Scatter plot with residual 
        RSPButton = Button(self, text = 'Residual', command = self.showImg4, bg='MediumPurple1')
        RSPButton.place(x = 500, y = 300)
        RSPLabel = Label(self, text = 'Display the residual on a scatter plot', font = 'Latha')
        RSPLabel.place(x = 10, y = 300)

        # Button that allows you to see all the calculations done on the data 
        DButton = Button(self, text = 'Display calculations', command = self.displayData, bg='MediumPurple1')
        DButton.place(x = 500, y = 400)
        DLabel = Label(self, text = 'Display calculations done in the terminal', font = 'Latha')
        DLabel.place(x = 10, y = 400)

        # Histogram with residual data 
        RButton = Button(self, text = 'Histogram', command = self.showImg5, bg='MediumPurple1')
        RButton.place(x = 500, y = 350)
        RLabel = Label(self, text = 'Histogram showing the residual values', font = 'Latha')
        RLabel.place(x = 10, y = 350)

        # Creating label 
        FLabel = Label(self, text = 'Enter your CSV file:', font = 'Latha')
        FLabel.place(x = 10, y = 100)

        def File(): 
            filename = e.get()
            print(filename)
            pd.read_csv(filename)
            return None

        # Creating Button for reading in a CSV file 
        CSVButton = Button(self, text = "Read CSV", command = self.readCSV, bg = 'MediumPurple1')
        CSVButton.place(x = 500, y = 10)

        # Creating label for the CSV button above 
        CSVLabel = Label(self, text = 'Choose the CSV file instead of manually typing it in', font = 'Latha')
        CSVLabel.place(x = 10, y = 10)
        
        # Creating text box 
        e = Entry(root, width = 30)
        e.pack()
        e.place(x = 200, y = 105)

        # Creating button for reading in a file
        FButton = Button(root, text = 'Enter', command = File, bg='MediumPurple1')
        FButton.place(x = 500, y = 100)

        # Creating button to display the r squared values for the test and train data sets 
        RButton = Button(self, text = "R-Squared", command = self.displayRSquared, bg = 'MediumPurple1')
        RButton.place(x = 500, y = 450)

        def file_save():
            # w in mode stands for write 
            name = asksaveasfile(mode='w', defaultextension = ".txt")
            text2save = str(textbx.get(2.0, END))
            name.write(text2save)
            name.close()

        # Creating button for saving input as a file 
        saveButton = Button(self, text = 'Save', command = file_save, bg = 'MediumPurple1')
        saveButton.place(x = 980, y = 450)

        # Text area 
        global textbx
        textbx = Text(root, height = 20, width = 80, bg = 'snow')
        textbx.place(x = 700, y = 100)

        # Creating clear button for text 
        clearButton = Button(self, text = 'Clear', command = self.clearContents, bg = 'MediumPurple1')
        clearButton.place(x = 980, y = 500)

        def showImg(self):
            xValue = e1.get()
            yValue = e2.get() 
            zValue = e3.get() 

            x_vars = [xValue, yValue] # cHECK ON GITHUB FOR THE VALUES 
            y_vars = zValue
            sns.pairplot(heart, x_vars, y_vars, size = 5, aspect = 1, kind = 'scatter')
            plt.show()

        # Scatter plot button and label
        SPButton = Button(self, text = 'Scatter Plot', command = self.showImg, bg='MediumPurple1')
        SPButton.place(x = 500, y = 150)
        SPLabel = Label(self, text = 'Display scatter plot', font = 'Latha')
        SPLabel.place(x = 10, y = 150)


        # DELETE IF NEEDED 
        # Creating text boxes 
        e1 = Entry(root, width = 10)
        e1.pack()
        e1.place(x = 700, y = 50)

        e2 = Entry(root, width = 10)
        e2.pack()
        e2.place(x = 800, y = 50)

        e3 = Entry(root, width = 10)
        e3.pack()
        e3.place(x = 900, y = 50)

        # CAN BE DELETED 
        # Creating label 
        varLabel = Label(self, text = 'Enter three varaibles that will be used in the graphs and other calc', font = 'Latha')
        varLabel.place(x = 10, y = 50)

        enterButton = Button(self, text = 'Enter', bg = 'MediumPurple1')
        enterButton.place(x = 1000, y = 50)


    def client_exit(self): 
        exit()  


 # pUT BACK 
    def showImg2(self): 
        plt.figure(figsize = (9, 9))
        sns.heatmap(heart.corr(), cmap = 'PuBu', annot=True)
        plt.xlabel('Values on x axis')
        plt.ylabel('Values on y axis')
        plt.show() 

    def showImg3(self): 
        plt.figure(figsize = (6, 6)) 
        plt.scatter(x_train, y_train)
        plt.plot(x_train, 201.157201 - 0.949434 * x_train, 'r')
        plt.show()

    def showImg4(self): 
        plt.figure(figsize = (6, 6)) 
        plt.scatter(x_train, res)
        plt.show() 

    def showImg5(self): 
        plt.figure(figsize = (9, 9)) 
        sns.distplot(res, bins=10)
        plt.title("Error", fontsize = 11)
        plt.xlabel('y_train - y_train_pred', fontsize = 10)
        plt.show()

    def readCSV(self): 
        from tkinter.filedialog import askopenfilename
        filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

        # Read in the given CSV file 
        pd.read_csv(filename)

    def displayData(self):
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
        
        textbx.insert('0.2', str(lr.summary())+ str(lr.params))

    def displayRSquared(self): 
        # Calculating r squared 
        from sklearn.metrics import r2_score 
        r_2_value = r2_score(y_train, y_train_pred)
        # Now dealing with the test data 

        # adding a constant to the test data for x 
        x_test_sm = model.add_constant(x_test)

        # predict 
        y_test_pred = lr.predict(x_test_sm)

        r_squared  = r2_score(y_test, y_test_pred)

        textbx.insert('0.2', '\nThis is the r squared value from the test data: ' + str(r_squared) + '\n\n'+ 'This is the r squared data from the train data: ' + str(r_2_value))
        print('\nThis is the r squared value from the test data: ')
        print(r_squared)
        print('\n')
        print('This is the r squared data from the train data: ')
        print(r_2_value)
        print('\n')
        
    def clearContents(self): 
        textbx.delete('0.2', 'end')
        
root = Tk()
root.attributes("-fullscreen", True)
root.geometry('600x900')
app = Window(root)
root.mainloop()
