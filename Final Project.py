#!/usr/bin/env python
# coding: utf-8

# In[1]:


#EDUBRIDGE CAPSTONE PROJECT
  #UBER DATA ANALYSIS
    
    
    
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gc
import os
import sys
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#!wget https://www.dropbox.com/s/ncqb2ctkg7da11k/weather.csv


# In[ ]:


#!wget https://www.dropbox.com/s/brixkogrmhan6ed/cab_rides.csv


# In[2]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


cab_data = pd.read_csv("cab_rides.csv")
#cab_data=reduce_mem_usage(cab_data)
weather_data = pd.read_csv("weather.csv")
#weather_data=reduce_mem_usage(weather_data)


# In[4]:


cab_data


# In[5]:


import datetime
cab_data['datetime']= pd.to_datetime(cab_data['time_stamp'])
cab_data
weather_data['date_time'] = pd.to_datetime(weather_data['time_stamp'])


# In[6]:


cab_data.columns


# In[7]:


weather_data.columns


# In[8]:


cab_data.shape


# In[9]:


weather_data.shape


# In[10]:


cab_data.describe()


# In[11]:


weather_data.describe()


# In[12]:


a=pd.concat([cab_data,weather_data])
a


# In[13]:


a['day']=a.date_time.dt.day
a['hour']=a.date_time.dt.hour


# In[14]:


a.fillna(0,inplace=True)


# In[15]:


a.columns
a


# In[16]:


a.cab_type.value_counts()


# In[17]:


a.groupby('cab_type').count().plot.bar()


# In[18]:


a['price'].value_counts().plot(kind='bar',figsize=(100,50),color='blue')


# In[19]:


a['hour'].value_counts().plot(kind='bar',figsize=(10,5),color='blue')


# In[20]:


import matplotlib.pyplot as plt
x=a['hour']
y=a['price']
plt.plot(x,y)
plt.show()


# In[21]:


x=a['rain']
y=a['price']
plt.plot(x,y)
plt.show()


# In[22]:


a.columns


# In[23]:


x1=a[['distance', 'temp', 'pressure', 'humidity','wind','rain','day','hour','surge_multiplier','clouds']]
y1=a['price']


# In[25]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.25, random_state = 42)


# In[26]:


linear=LinearRegression()
linear.fit(x_train,y_train)


# In[27]:


predictions=linear.predict(x_test)


# In[28]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
df


# In[29]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(26,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:




