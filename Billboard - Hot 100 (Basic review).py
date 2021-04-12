#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # Hides warning


# In[2]:


df = pd.read_csv('charts.csv')


# In[3]:


df = df.fillna(0)
df.head()


# In[4]:


artistdf = df['artist'].value_counts().head(10)
print(artistdf)


# In[5]:


songdf = df['song'].value_counts().head(10)
print(songdf)


# In[9]:


print(df[["artist", "song", "peak-rank"]].value_counts().head(10))


# In[10]:


df['date'] = df['date'].apply(pd.to_datetime)
df.set_index('date', inplace=True)
df.head()


# In[11]:


# df.resample('A')['artist'].agg(['first','last']) # It can be used to find the most and least listened ones.
df.resample('A').first()


# In[12]:


artistdf = df['artist'].value_counts().head(10)
print(artistdf)


# In[14]:


songdf = df['song'].value_counts().head(10)
print(songdf)


# In[ ]:




