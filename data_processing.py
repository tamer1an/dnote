
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


dataset = pd.read_csv('./Data.csv')


# In[4]:


print(dataset)


# #### y - inspected value
# #### x - data model

# In[18]:


x = dataset.iloc[:,:-1].values


# In[19]:


y = dataset.iloc[:,-1].values


# #### Transforming missing values

# In[20]:


from sklearn.preprocessing import Imputer


# In[23]:


imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)


# In[24]:


cleanResult = imputer.fit(x[:, 1:3])


# In[25]:


x[:, 1:3] = cleanResult.transform(x[:, 1:3])


# In[34]:


print(x)


# #### Transforming text to index

# In[39]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[28]:


labelEncoder_X = LabelEncoder()


# In[36]:


x[:, 0] = labelEncoder_X.fit_transform(x[:,0])


# In[37]:


print(x)


# #### Transformin indexex to columns with 1 & 0

# In[42]:


oneHotEncoder = OneHotEncoder(categorical_features=[0])


# In[44]:


x = oneHotEncoder.fit_transform(x).toarray()


# In[46]:


print(y)


# In[47]:


labelEncoder_Y = LabelEncoder()


# In[48]:


y = labelEncoder_Y.fit_transform(y)


# In[49]:


print(y)


# #### Splitting dataset Training set and Test set

# In[50]:


from sklearn.model_selection import train_test_split


# In[52]:


x_traint, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# #### Feature scaling

# In[54]:


from sklearn.preprocessing import StandardScaler


# In[55]:


sc_x = StandardScaler()


# In[56]:


x_traint = sc_x.fit_transform(x_traint)


# In[57]:


x_test = sc_x.transform(x_test)


# _Dummy variables scale and lose identity?_
