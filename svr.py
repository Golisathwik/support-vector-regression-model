import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

data= pd.read_csv('data.csv')
model= SVR(kernel='rbf', C=100, epsilon=0.1) #change to linear, poly and rbf to see the difference
x_scale=StandardScaler()
y_scale=StandardScaler()
x_data= data.iloc[:,0].values
y_data= data.iloc[:,1].values
x= x_scale.fit_transform(x_data.reshape(-1,1))
y= y_scale.fit_transform(y_data.reshape(-1,1)).ravel()
model.fit(x,y)
predict_value= model.predict(x_scale.transform([[3.6]]))
value= y_scale.inverse_transform(predict_value.reshape(-1,1)).ravel()
print(f"the salary for 3.6 years of experience is {int(value[0])}")
x_predict= model.predict(x)
plt.scatter(x_data,y_data, color='red')
plt.plot(x_data,y_scale.inverse_transform(x_predict.reshape(-1,1)), color= 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()