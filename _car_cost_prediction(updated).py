#!/usr/bin/env python
# coding: utf-8

# In[173]:


# Importing all the libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings 
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[174]:


# creating the object of the classes
enc=LabelEncoder()
regressor=RandomForestRegressor()


# In[175]:


# importing all the data
train_data=pd.read_excel("C:/Users/RAGHAVENDRA/Desktop/verzeo/Data_Train.xlsx")
test_data=pd.read_excel("C:/Users/RAGHAVENDRA/Desktop/verzeo/Data_Test.xlsx")


# # Having a glance
# 

# In[176]:


train_data.head()


# In[177]:


test_data.head()


# # Cleansing the train_data

# In[178]:


# cleansing the data includes->converts a)int and float values to their respective mean of the coumn values, and string to null string 
d=train_data.dtypes # encatches the data type train_data
print("Before data cleansing\n\n\n\n")
print(train_data.info()) # gets the count of non-Null values in each and every columns of dataframw 



s=train_data.shape #s holds the dimension of the dataframe,where s[0]->number of rows,s[1]->number of columns

for i in range(s[1]): #This loops through all the columns of the dataframe  
    if train_data.iloc[:,i].isnull().sum()>0:# this condition executes if particular column has any null values,that's why isnull().sum() is used
        if d[i]=="int64" or d[i]=="float64": # if in case the datatype of the column is float or int
            train_data.iloc[:,i].fillna(value=np.mean(train_data.iloc[:,i]),inplace=True)#the null values will be filled with the mean of the rsepective column
        elif d[i]=="object":# if the datatype of the column is object , then
            train_data.iloc[:,i].fillna(value="",inplace=True)# null vaues will be filled with  null string,
            
print("\n\n\nData is now cleaned\n\n\n")
print(train_data.info())#now checking, whether every null values are replaced or not


# # cleansing the test data

# In[179]:


#explaination is already told in the above section ,for cleaning the dataset of train_data

d=test_data.dtypes
print("Before data cleansing\n\n\n\n")
print(test_data.info())



s=test_data.shape
for i in range(s[1]):
    if test_data.iloc[:,i].isnull().sum()>0:
        if d[i]=="int64" or d[i]=="float64":
            test_data.iloc[:,i].fillna(value=np.mean(test_data.iloc[:,i]),inplace=True)
        elif d[i]=="object":
            test_data.iloc[:,i].fillna(value="",inplace=True)
print("\n\n\nData is now cleaned\n\n\n")
print(test_data.info())


# # Combining  both train data and test data,to apply faeture engineering techniques,so that Label Encoder produces exact Encoded values and converting the data becomes easy.

# In[180]:


data=pd.concat([train_data,test_data],axis=0)


# In[181]:


data.shape


# In[182]:


data.head()


# In[183]:


data.tail()


# # Everything ,we feed to model must be of numeric type: 1)Integer 2)Float,we require to apply feature engineering here.
# 
# # Engine,Power,Milage should not be encoded through LabelEncoder, because we need to fit the model in regressor mode,if we use the encoding fromat,it will obviously fit the model in wrong way,which will obviously predict the wrong values.
# 
# 
# 
# # Engine 

# In[184]:


eng=[]
for i in list(data["Engine"]):
    if len(i)>3:
        eng.append(float(i[:-3]))
    elif i=="null":
        eng.append(np.average(eng))
    else:
        eng.append(np.average(eng))
    
data["Engine"]=eng
data["Engine"]=data["Engine"].astype(float)


# # Power

# In[185]:


powe=[]
for i in list(data["Power"]):
    
    if i=="null bhp" or i=="null ":
        powe.append(np.average(powe))
    elif len(str(i))>3:
        powe.append(float(i[:-3]))
    else:
        powe.append(np.average(powe))



data["Power"]=powe

data["Power"]=data["Power"].astype(float)


# # Mileage

# In[186]:


mil=[]
for i in list(data["Mileage"]):
    i=str(i)
    if len(i)>0:
        if i[-1]=="g":
            mil.append(i[:-6])
        elif i[-1]=="l":
            mil.append(i[:-5])
    else:
        mil.append(0)

data["Mileage"]=mil

data["Mileage"]=data["Mileage"].astype(float)


# ### checking whether datatypes of milage,power,engine has converted to the float types from object,as shown:

# In[187]:


data.dtypes


# # preview of newly converted data

# In[189]:


data.head()


# # Name,Location,Fuel_type,Transmission,Owner_Type:  For these Label Encoder can be used,to fit the model we convert string to int or float,the best way to convert is  label Encoder.
# 
# # usage of label encoder

# In[190]:


col=data.columns #Capture the column names of the datarame 

for i in col: # going through all the names of the column
    if data[i].dtypes=="O": # if in case the data type of the column is object,
        data[i]=enc.fit_transform(data[i]) #then it is converted into the respective encoding format


# # data preview and data types:

# In[191]:


data.head()


# In[192]:


data.dtypes


# In[193]:


# getting the size of train_data,which is helpful to predict.
train_data.shape


# In[194]:


test_data.shape


# # collecting the data after cleaning and converting to the required format

# In[195]:


x=data.iloc[:6019,:-1]
y=data.iloc[:6019,-1]
x_test=data.iloc[6019:,:-1]


# # Using Decion tree Regressor and Random Forest Regressor ,will predict the values far better than Linear Regression.
# 
# 
# # But Random forest is more good than the decison Tree regressor,because the Random forest by default has "n_estimators=100", number of Decision tree produed under this is 100,so result predicted by this is too good than Desion tree.that's why I chose RandomForestRegressor as my prediction model.
# 
# # After analysing predictions' with train data ,I chose RandomForestRegressor

# In[196]:



regressor=RandomForestRegressor()
regressor.fit(x,y)
pr=regressor.predict(x_test)
test_data["Price_Random"]=pr


# In[197]:


test_data.head()

