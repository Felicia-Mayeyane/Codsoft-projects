#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


data=pd.read_csv('/Users/feliciasmac/Downloads/IRIS.csv')


# In[23]:


print(data)


# In[24]:


data.shape


# In[25]:


data.describe


# In[26]:


data.isnull().sum() #no missing value


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris


# In[29]:


# Plot the species distribution 
data['species'].value_counts().plot(kind='bar')
plt.title('Iris Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()


# In[32]:


X = data.drop('species', axis=1)
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[33]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[34]:


y_pred = model.predict(X_test)


# In[37]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




