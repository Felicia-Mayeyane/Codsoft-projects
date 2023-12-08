#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[51]:


data= pd.read_csv("/Users/feliciasmac/Downloads/creditcard.csv")


# In[53]:


print(data)


# In[55]:


data.shape


# In[57]:


data.head(20)


# In[59]:


data.info


# In[61]:


print(data.describe())


# In[35]:


print(data['Class'].value_counts())


# In[36]:


print(data.isnull().sum())


# In[37]:


sns.countplot(data['Class'])
plt.title('Class Distribution')
plt.show()


# In[38]:


X = data.drop('Class', axis=1)
y = data['Class']


# In[39]:


X = data.drop('Class', axis=1)
y = data['Class']


# In[40]:


# Apply SMOTE for oversampling
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)


# In[41]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[42]:


# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[45]:


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[47]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




