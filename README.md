# ENTRY-EXAM-INTO-UNIV-PERFORMANCE
Data for JAMB exam in Maths, English and Biology was collected for both Male and Female students across 6 states seeking admission into Three major Universities.UI, UNILAG and OAU
#import pandas 
import pandas as pd
#import seaborn module
import seaborn as sns
# set up the matplotlib environment
import matplotlib.pyplot as plt
#EXPLORING THE DATA
dataframe = pd.read_csv("StudentsJambResult.csv")
dataframe.head()
#DATA VISUALIZATION
# USING THE COUNT PLOT
sns.set(rc={'figure.figsize':(10.7,11.2)})
sns.countplot(x='gender', hue='university of choice', data=dataframe)
plt.show()
#the distribution shows more females chose UNILAG AND UI while More male chose OAU
sns.set(rc={'figure.figsize':(10.7,11.2)})
sns.violinplot(x='biology score',y='gender', hue='gender', data=dataframe)
plt.show()
#the distribution shows that more male had higher scores than female but a female scored highest in biology

sns.set(rc={'figure.figsize':(10.7,11.2)})
sns.violinplot(x='math score',y='gender', hue='gender', data=dataframe)
plt.show()
#The distribution shows that maths result was at par but male scored higher average

#TO EVALUATE THE DATA USING RANDOM FOEST REGRESSOR
# import necessary module
from sklearn import preprocessing
# set up label encoder object
le = preprocessing.LabelEncoder()
#convert columns into numerics using the label encoder
dataframe["gender"] = le.fit_transform(dataframe["gender"])
dataframe["university of choice"] = le.fit_transform(dataframe["university of choice"])
dataframe["state"] = le.fit_transform(dataframe["state"])
dataframe["parental level of education"] = le.fit_transform(dataframe["parental level of education"])
dataframe["test preparation course"] = le.fit_transform(dataframe["test preparation course"])
# To stat Training, first of all, divide data into attributes and tables
X = dataframe.iloc[:, 0:7].values
y = dataframe.iloc[:, 7].values
#let's divide data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
import numpy as np
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test) 
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
Mean Absolute Error: 3.556896551724138
Mean Squared Error: 19.94318965517242
Root Mean Squared Error: 4.465779848489222
#The root mean square error is 4.46577 which is less than 10% of the average score. This was done with Nn20
#n=80 and were in the range 4.4 and 4.5. The performance was a good performance. It is predictable that most of these students will gain admission in the university
