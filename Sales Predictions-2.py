#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('/Users/feliciasmac/Downloads/advertising.csv')


# In[3]:


print(data.describe())


# In[4]:


print(data.isnull().sum())


# In[5]:


print(data.head(10))


# In[6]:


sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='scatter')
plt.show()


# In[25]:


# Create a new column for total sales
data['Total_Sales'] = data['TV'] + data['Radio'] + data['Newspaper']

# Display the updated dataset
print(data)


# In[17]:


data['TotalSales'] = data['TV'] + data['Radio'] + data['Newspaper']


# In[18]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[19]:


X = data.drop('Sales', axis=1)
y = data['Sales']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# In[12]:


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')


# In[28]:


# Visualize predictions vs actuals
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Predictions vs Actuals')
plt.show()


# In[27]:


# Visualize feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Features')
plt.show()


# In[ ]:




