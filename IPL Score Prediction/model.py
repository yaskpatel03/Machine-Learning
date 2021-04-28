#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


df = pd.read_csv("new_data.csv")


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df = df.drop(df.columns[[0]], axis=1) 


# In[9]:


df.info()


# In[10]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[11]:


from sklearn import metrics , model_selection

## Import the Classifier.
from sklearn.naive_bayes import GaussianNB


# In[12]:


# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=123)


# In[13]:


model = GaussianNB()
## Fit the model on the training data.
model.fit(X_train, y_train)


# In[14]:


# use the model to make predictions with the test data
y_pred = model.predict(X_test)
# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[ ]:





# In[ ]:





# In[1]:


from sklearn.svm import SVC
model = SVC( kernel ='linear')


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# In[ ]:





# In[15]:


from sklearn.model_selection import GridSearchCV


# In[16]:


param={'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
      'C':np.arange(1,42,10),
      'degree':np.arange(3,6),   
      'coef0':np.arange(0.001,3,0.5),
      'gamma': ('auto', 'scale')}


# In[ ]:


SVModel = SVC()
GridS = GridSearchCV(SVModel, param, cv=5)
GridS.fit(X_train, y_train)


# In[ ]:


GridS.best_params_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(df[['venue','innings','ball','batting_team','bowling_team','no_of_wickets']],df['total_run'])


# In[ ]:


thresold = 1000000
df.head()


# In[ ]:


# Splitting the data into train and test set
X_train = df.drop(labels='total_run', axis=1)[df['match_id'] <= thresold]
X_test = df.drop(labels='total_run', axis=1)[df['match_id'] >= thresold ]

y_train = df[df['match_id'] <= thresold]['total_run'].values
y_test = df[df['match_id'] >= thresold ]['total_run'].values


# In[ ]:


X_train.drop(labels='match_id', axis=True, inplace=True)
X_test.drop(labels='match_id', axis=True, inplace=True)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[ ]:


X_test.info()


# In[ ]:


regressor.predict([[0,5,1,3.4,6,8,0]])


# In[ ]:




