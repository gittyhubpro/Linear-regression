#The following scatter plots of heart diseases occurences with age and cholestrol playing a factor.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#to read CSV files.
heart_disease = pd.read_csv("[address file of heart.csv]")
print(np.shape(heart_disease))      #shape
print(heart_disease.head())       #head


#plotting the scatter to understand
heart_disease.plot.scatter("age","chol")

#training the model by splitting data(80,20)
x_train, x_test, y_train, y_test = train_test_split(heart_disease.age, heart_disease.chol, test_size = 0.2)

#linear_regression_creation and execution

line_reg = LinearRegression()
line_reg.fit(np.array(x_train).reshape(-1,1),y_train)
pred = line_reg.predict(np.array(x_test).reshape(-1,1))
residual =  pred - y_test 

ms = mean_squared_error(y_test, pred)**0.5

#plotting with regression line.

plt.scatter(heart_disease.age,heart_disease.chol)
plt.plot(x_test,pred,color = "red")
plt.show()
