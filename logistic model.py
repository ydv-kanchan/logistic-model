#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import load_dataset
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[4]:


data = load_dataset("titanic")
data


# In[5]:


data.info()


# In[6]:


columns = ['alive', 'alone', 'embark_town', 'who', 'adult_male', 'deck']
data_2 = data.drop(columns, axis=1)


# In[7]:


data_2.describe(include='all').T


# In[9]:


print(f"Max value of age column : {data_2['age'].max()}")
print(f"Min value of age column : {data_2['age'].min()}")


# In[10]:


bins = [0, 5, 17, 25, 50, 80]
labels = ['Infant', 'Kid', 'Young', 'Adult', 'Old']
data_2['age'] = pd.cut(data_2['age'], bins = bins, labels=labels)


# In[17]:


pd.DataFrame(data_2['age'].value_counts())


# In[19]:


data_2['age'].mode()[0]


# In[23]:


data_4 = data_2.fillna({'age' : data_2['age'].mode()[0]})


# In[24]:


data_2['embarked'].unique()


# In[25]:


print(f"How many 'S' on embarked column : {data_2[data_2['embarked'] == 'S'].shape[0]}")


# In[26]:


print(f"How many 'S' on embarked column : {data_2[data_2['embarked'] == 'S'].shape[0]}")


# In[27]:


print(f"How many 'Q' on embarked column : {data_2[data_2['embarked'] == 'Q'].shape[0]}")


# In[28]:


data_3 = data_2.fillna({'embarked' : 'S'})


# In[29]:


data_4[['pclass', 'survived']].groupby(['pclass']).sum().sort_values(by='survived')


# In[30]:


data_4[['sex', 'survived']].groupby(['sex']).sum().sort_values(by='survived')


# In[36]:


bins = [-1, 7.9104, 14.4542, 31, 512.330]
labels = ['low', 'medium-low', 'medium', 'high']
data_4['fare'] = pd.cut(data_4["fare"], bins = bins, labels = labels)


# In[37]:


data_5 = data_4.drop('class', axis=1)
sns.distplot(data_5['survived'])


# In[38]:





# In[39]:





# In[40]:


plt.figure(figsize=(20, 10))
plt.subplot(321)
sns.barplot(x = 'sibsp', y = 'survived', data = data_5)
plt.subplot(322)
sns.barplot(x = 'fare', y = 'survived', data = data_5)
plt.subplot(323)
sns.barplot(x = 'pclass', y = 'survived', data = data_5)
plt.subplot(324)
sns.barplot(x = 'age', y = 'survived', data = data_5)
plt.subplot(325)
sns.barplot(x = 'sex', y = 'survived', data = data_5)
plt.subplot(326)
sns.barplot(x = 'embarked', y = 'survived', data = data_5);


# In[41]:


dummies = ['fare', 'age', 'embarked', 'sex']
dummy_data = pd.get_dummies(data_5[dummies])


# In[42]:


dummy_data.shape


# In[43]:


data_6 = pd.concat([data_5, dummy_data], axis = 1)
data_6.drop(dummies, axis=1, inplace=True)


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# In[45]:


X = data_6.drop('survived', axis = 1)
y = data_6['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# In[46]:


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_pred


# In[47]:


accuracy_score(y_pred, y_test)


# In[48]:


confusion_matrix(y_pred, y_test)


# In[ ]:




