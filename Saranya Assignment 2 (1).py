#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Load the dataset.

# In[7]:


df=pd.read_csv('Downloads\Churn_Modelling.csv')
df.head()


# In[10]:


df.shape


# In[12]:


df.info()


# In[13]:


df.head()


# # Multi - Variate Analysis

# In[16]:


df.plot()


# # Univariate Analysis

# In[18]:


df.Exited.plot()


# In[19]:


df.RowNumber.plot()


# # Bi - Variate Analysis

# In[22]:


df.Exited.plot()
df.RowNumber.plot()
plt.legend(['Exited','RoeNumber'])


# # Perform descriptive statistics on the dataset.

# In[39]:


df=pd.read_csv('Downloads\Churn_Modelling.csv')
df.sum()


# In[44]:


df.sum(1)


# In[43]:


df.mean()


# In[37]:


df.describe()


# In[40]:


df.prod()


# In[41]:


df.describe(include=['object'])


# In[42]:


df. describe(include='all')


# # Handle the Missing values.

# In[45]:


df.isnull()


# In[55]:


df.notnull()


# In[61]:


df=pd.read_csv('Downloads\Churn_Modelling.csv')
bool_series = pd.isnull(df["Surname"])
df[bool_series]


# In[56]:


df.fillna(0)


# In[57]:


df.fillna(method ='pad')


# In[59]:


df=pd.read_csv('Downloads\Churn_Modelling.csv')
df[10:25]


# # Find the outliers and replace the outliers

# In[108]:


sns.boxplot(df.Age)


# In[109]:


df.shape


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


dataframe=pd.read_csv('Downloads\Churn_Modelling.csv')
dataframe.head()


# In[20]:


q1 = dataframe.Age.quantile(0.25)
q3 = dataframe.Age.quantile(0.75)


# In[140]:


IQR=q3-q1


# In[45]:


upper_limit= q3+1.5*IQR
lower_limit= q1-1.5*IQR


# In[46]:


dataframe=dataframe[dataframe.Age<upper_limit]


# In[111]:


sns.boxplot(dataframe.Age)


# In[112]:


df.shape


# # Replacement of outliers-median

# In[103]:


sns.boxplot(df.RowNumber)


# In[47]:


q1 = dataframe.RowNumber.quantile(0.25)
q3 = dataframe.RowNumber.quantile(0.75)


# In[ ]:


IQR=q3-q1


# In[ ]:


upper_limit= q3+1.5*IQR
lower_limit= q1-1.5*IQR


# In[48]:


upper_limit


# In[82]:


df.median()


# In[101]:


df['RowNumber'] = np.where(df['RowNumber']>upper_limit,5.0,df['RowNumber'])


# In[102]:


sns.boxplot(df.RowNumber)


# In[104]:


df.shape


# # Z-score

# In[310]:


sns.boxplot(dataframe.Age)


# In[34]:


from scipy import stats


# In[38]:


Age_zscore=stats.zscore(df.Age)


# In[39]:


Age_zscore


# In[36]:


df_z=df[np.abs(Age_zscore)<=3]


# In[313]:


sns.boxplot(df_z['Age'])


# In[63]:


df.shape


# # outlier removel with percentile

# In[171]:


sns.boxplot(dataframe.RowNumber)


# In[98]:


p99=df.RowNumber.quantile(0.99)
p99


# In[100]:


df.describe()


# In[ ]:


df=df[df.RowNumber<=p99]


# In[113]:


sns.boxplot(df.RowNumber)


# In[114]:


df.shape


# # Check for Categorical columns and perform encoding.

# In[ ]:


# categorical columns


# In[304]:


import pandas as pd
import numpy as np
import random

df = pd.DataFrame({
    'x': np.linspace(0, 50, 6),
    'y': np.linspace(0, 20, 6),
    'Age': random.sample('abcdef', 6)
})
df['Age'] = pd.Categorical(df['Age'])


# In[305]:


df.Age.dtype


# In[306]:


df.x.dtype == 'float64'


# In[ ]:


# 1 label encoding


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le=LabelEncoder()


# In[ ]:


df.sex=le.fit_transform(df.sex)
df.smoker=le.fit_transfrom(df.smoker)


# In[115]:


df.head()


# In[117]:


#2 One hot encoding


# In[119]:


df_main=pd.get_dummies(df,columns=['Age'])
df_main.head()


# In[120]:


df_main.corr()


# # Split the data into dependent and independent variables.

# In[121]:


# X and Y split


# In[127]:


# dependent variable

Y=df_main['RowNumber']
Y


# In[128]:


# independent variable

X=df_main.drop(columns=['RowNumber'],axis=1)
X.head()


# # Scale the independent variables

# In[297]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
X = pd.DataFrame(X)
sc= StandardScaler()


# In[299]:


X_scaled=pd.DataFrame(scale(X),columns=X.columns)
X_scaled.head()


# # Split the data into training and testing

# In[296]:


#Train Test Split


# In[293]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[268]:


from sklearn.model_selection.train_test_split(*arrays, **options)


# In[245]:


X = np.arange(1,25).reshape(12,2)
Y = np.array([0,1,1,0,1,0,0,1,1,0,1,0])
X


# In[246]:


Y


# In[248]:


X_train,X_test,Y_train,y_test = train_test_split(X,Y)
X_train


# In[249]:


X_test


# In[250]:


Y_train


# In[253]:


y_test


# In[257]:


X_train,X_test, Y_train,Y_test =train_test_split(X,Y,test_size=5, random_state=4)


# In[256]:


X_train


# In[260]:


X_test


# In[261]:


Y_train


# In[262]:


Y_test


# In[263]:


X_train,X_test, Y_train,Y_test =train_test_split(X,Y,test_size=0.33, random_state=4,stratify=Y)


# In[264]:


X_train


# In[265]:


X_test


# In[266]:


Y_train


# In[267]:


Y_test

