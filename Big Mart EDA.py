#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


f1=pd.read_csv("Train.csv")


# In[3]:


f2=pd.read_csv("Test.csv")


# In[4]:


frames=[f1,f2]
df = pd.concat(frames, sort ='False')


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


dfn=df
dfp=df


# In[9]:


df


# In[ ]:





# In[10]:


df.columns = map(str.lower, df.columns)


# In[11]:


sns.catplot('item_fat_content',kind = 'count',data = df,aspect =4)


# In[12]:


sns.catplot('item_type',kind = 'count',data = df,aspect =4)


# In[13]:


sns.catplot('outlet_size',kind = 'count',data = df,aspect =4)


# In[14]:


sns.catplot('outlet_type',kind = 'count',data = df,aspect =4)


# In[15]:


corr = df.corr()
corr = corr.sort_values('item_mrp', ascending=False)
plt.figure(figsize=(8,10))
sns.barplot( corr.item_mrp[1:], corr.index[1:], orient='h')
plt.show()


# In[16]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(df)


# In[ ]:





# In[17]:


sns.boxplot(data=df, orient="h", palette="Set2")


# In[ ]:





# In[18]:


sns.barplot(x="item_mrp", y="item_fat_content", palette="rocket", data=df)


# In[19]:


plt.figure(figsize=(16, 6))
figure=sns.barplot(y="item_mrp", x="item_type", palette="rocket", data=df)
figure.set_xticklabels(figure.get_xticklabels(), rotation=45, horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')


# In[20]:


df.boxplot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[22]:


df = remove_outlier(df,'item_outlet_sales')


# In[23]:


sns.boxplot(data=df, orient="h", palette="Set2")


# In[24]:


replace_map = {'item_fat_content': {'Low Fat': 1, 'low fat': 1, 'Regular': 2, 'LF': 1,'reg': 2}}


# In[25]:


df['item_fat_content'].replace(replace_map, inplace=True)


# In[26]:


df['item_fat_content']= df['item_fat_content'].replace(to_replace=['Low Fat', 'low fat','Regular', 'LF', 'reg'], value=[1,1,2,1,2])


# In[27]:


df


# In[28]:


dfn


# In[29]:


dfn['item_fat_content']= dfn['item_fat_content'].replace(to_replace=['Low Fat', 'low fat','Regular', 'LF', 'reg'], value=[1,1,2,1,2])


# In[30]:


dfn['outlet_size']= dfn['outlet_size'].replace(to_replace=['Small','Medium','High'], value=[1,2,3])


# In[31]:


dfn['outlet_location_type']= dfn['outlet_location_type'].replace(to_replace=['Tier 1','Tier 2','Tier 3'], value=[1,2,3])


# In[32]:


dfn = dfn.drop(['item_identifier', 'outlet_identifier'], axis=1)


# In[ ]:





# In[33]:


df3=dfn
df4=dfn


# In[34]:


dfn


# In[35]:


sns.barplot(y="item_mrp", x="outlet_type", palette="rocket", data=df)


# In[36]:


dfn = dfn.drop(['outlet_type'], axis=1)


# In[37]:


dfn


# In[38]:


count_nan = len(dfn) - dfn.count()
count_nan


# In[39]:


df3=dfn
df4=dfn


# In[40]:


dfn = dfn.dropna() 


# In[41]:


dfn


# In[42]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(dfn[['item_type']]).toarray())
# merge with main df bridge_df on key values
dfn = dfn.join(enc_df)
dfn


# In[43]:


one_hot = pd.get_dummies(df3['item_type'])
# Drop column B as it is now encoded
df3 = df3.drop('item_type',axis = 1)
# Join the encoded df
df3 = df3.join(one_hot)
df3  


# In[44]:


df3.info()


# In[45]:


df3=df3.dropna()


# In[46]:


df3.info()


# In[47]:


#Logistic regression


# In[48]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score


# In[55]:


x= df3['item_mrp']
y= df3.drop('item_mrp',axis=1)

x=x.reshape(-1,1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.30, 
                                                    random_state=42)


# In[ ]:


regressor=LinearRegression()
X=X.reshape(-1,1)
regressor.fit(X,y)


# In[53]:


from sklearn.linear_model import LinearRegression as lm
model=lm().fit(X_train,y_train)
predictions=model.predict(X_test)
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)


# In[56]:


x


# In[57]:


y

