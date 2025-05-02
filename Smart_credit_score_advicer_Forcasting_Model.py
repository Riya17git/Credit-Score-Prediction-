#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle


# In[2]:


card = pd.read_csv('cards_data.csv')
transaction = pd.read_csv('transactions_data.csv')
user = pd.read_csv('users_data.csv')


# In[3]:


transaction = transaction.sample(frac=0.5, random_state=42)


# In[4]:


int_cols = ['id','client_id','cvv','num_cards_issued','year_pin_last_changed']
float_cols = ['card_number']

card[int_cols] = card[int_cols].astype('int32')
card[float_cols] = card[float_cols].astype('float32')

print(card.info())


# In[5]:


int_cols_transaction = ['id','client_id','card_id','merchant_id','mcc']
float_cols_transaction = ['zip']

transaction[int_cols_transaction] = transaction[int_cols_transaction].astype('int32')
transaction[float_cols_transaction] = transaction[float_cols_transaction].astype('float32')

print(transaction.info())


# In[6]:


int_cols_user = ['id','current_age','retirement_age','birth_year','credit_score','num_credit_cards','birth_month']
float_cols_user = ['latitude','longitude']

user[int_cols_user] = user[int_cols_user].astype('int32')
user[float_cols_user] = user[float_cols_user].astype('float32')

print(user.info())


# ## Importing mcc_codes and placing in transaction table

# In[7]:


with open('mcc_codes.json') as f:
  mcc_code = json.load(f)


# In[8]:


transaction['mcc'] = transaction['mcc'].astype(str)


# In[9]:


transaction['mcc_description'] = transaction['mcc'].map(mcc_code)


# # # Merging the dataset

# In[10]:


merged_df = pd.merge(transaction,card, on='client_id', how = 'inner')
merged_df.head()


# In[11]:


merged_data = pd.merge(merged_df,user, left_on='client_id', right_on='id', how = 'inner')
merged_data.head()


# # Clean the Data 

# In[12]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[13]:


merged_data.head()


# In[14]:


merged_data = merged_data.drop_duplicates()


# #  Cleaning Data

# In[15]:


merged_data = merged_data.rename(columns={'id_x':'id_transaction','id_y':'id_card'})


# In[16]:


merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data['expires'] = pd.to_datetime(merged_data['expires'],format='%m/%Y')
merged_data['acct_open_date'] = pd.to_datetime(merged_data['acct_open_date'],format='%m/%Y')


# In[17]:


merged_data['amount'] = merged_data['amount'].replace('[\$,]', '', regex=True).astype(float)
merged_data['credit_limit'] = merged_data['credit_limit'].replace('[\$,]', '', regex=True).astype(float)
merged_data['per_capita_income'] = merged_data['per_capita_income'].replace('[\$,]', '', regex=True).astype(float)
merged_data['yearly_income'] = merged_data['yearly_income'].replace('[\$,]', '', regex=True).astype(float)
merged_data['total_debt'] = merged_data['total_debt'].replace('[\$,]', '', regex=True).astype(float)


# In[18]:


merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data['year'] = merged_data['date'].dt.year
merged_data['month'] = merged_data['date'].dt.month
merged_data['day'] = merged_data['date'].dt.day


# In[19]:


merged_data = merged_data.dropna(subset=['merchant_state', 'zip', 'errors'])


# In[20]:


merged_data.shape


# In[21]:


merged_data.head()


# In[22]:


label_column = ['merchant_city','merchant_state','mcc_description']
one_hot_column = ['use_chip','card_brand','card_type','has_chip','gender','card_on_dark_web']

label_encoder = LabelEncoder()
for col in label_column:
    merged_data[col] = label_encoder.fit_transform(merged_data[col])
    
merged_data = pd.get_dummies(merged_data, columns=one_hot_column)


# In[23]:


merged_data = merged_data.astype(int, errors='ignore')


# In[24]:


merged_data = merged_data.drop('address',axis='columns')
merged_data = merged_data.drop('card_number',axis='columns')
merged_data = merged_data.drop('errors',axis='columns')


# In[25]:


merged_data.shape


# #  Handling Expires and acct_open_date

# In[26]:


merged_data['acct_open_year'] = merged_data['acct_open_date'].dt.year
merged_data['acct_open_month'] = merged_data['acct_open_date'].dt.month
merged_data['acct_open_date'] = merged_data['acct_open_date'].dt.day


# In[27]:


merged_data = merged_data.drop('acct_open_date',axis='columns')


# In[28]:


merged_data['expires_year'] = merged_data['expires'].dt.year
merged_data['expires_month'] = merged_data['expires'].dt.month
merged_data['expires_date'] = merged_data['expires'].dt.day


# In[29]:


merged_data = merged_data.drop('expires',axis='columns')


# In[30]:


merged_data = merged_data.drop('date',axis='columns')


# In[31]:


merged_data['mcc'] = merged_data['mcc'].astype(int)


# In[ ]:





# In[ ]:





# In[32]:


merged_data.head()


# In[33]:


print(merged_data['year'].unique())


# In[34]:


print(merged_data['month'].unique())


# In[35]:


merged_data.info()


# In[36]:


merged_data = merged_data.sort_values(by=['client_id', 'year', 'month'])


# In[37]:


merged_data['credit_score_3m'] = merged_data.groupby('client_id')['credit_score'].shift(-3)
merged_data['credit_score_6m'] = merged_data.groupby('client_id')['credit_score'].shift(-6)


# In[38]:


merged_data.head(5)


# In[39]:


merged_data.shape


# In[40]:


from sklearn.model_selection import train_test_split

selected_features = ['credit_limit', 'yearly_income', 'total_debt', 'num_credit_cards', 'current_age','birth_year','birth_month']
X = merged_data[selected_features]
y_3m = merged_data['credit_score_3m']
y_6m = merged_data['credit_score_6m']

X_train, X_test, y_train_3m, y_test_3m = train_test_split(X, y_3m, test_size=0.2, random_state=42)


# In[41]:


X_train, X_test, y_train_6m, y_test_6m = train_test_split(X, y_6m, test_size=0.2, random_state=42)


# #  3 months prediction

# In[42]:


# Filter out nulls in y_train_3m
mask_train = ~y_train_3m.isnull()  # True for non-null entries

X_train_clean = X_train[mask_train]
y_train_3m_clean = y_train_3m[mask_train]


# In[43]:


# Filter out nulls in y_test_3m
mask_test = ~y_test_3m.isnull()

X_test_clean = X_test[mask_test]
y_test_3m_clean = y_test_3m[mask_test]


# In[44]:


print(X_train_clean.shape)
print(X_test_clean.shape)
print(y_train_3m_clean.shape)
print(y_test_3m_clean.shape)


# In[45]:


from sklearn.ensemble import RandomForestRegressor

# Create two separate models
model_3m = RandomForestRegressor(n_estimators=100, random_state=42)
model_6m = RandomForestRegressor(n_estimators=100, random_state=42)

# Train them on their respective targets
model_3m.fit(X_train_clean, y_train_3m_clean)
#model_6m.fit(X_train, y_train_6m)


# In[46]:


from sklearn.metrics import mean_squared_error, r2_score

y_pred_3m = model_3m.predict(X_test_clean)


# In[47]:


print("3-Month Forecast RMSE:", mean_squared_error(y_test_3m_clean, y_pred_3m, squared=False))


# In[48]:


print("3-Month R²:", r2_score(y_test_3m_clean, y_pred_3m))


# #  6 months prediction

# In[49]:


# Filter out nulls in y_train_6m
mask_train = ~y_train_6m.isnull()  # True for non-null entries

X_train_clean = X_train[mask_train]
y_train_6m_clean = y_train_6m[mask_train]


# In[50]:


# Filter out nulls in y_test_6m
mask_test = ~y_test_6m.isnull()

X_test_clean = X_test[mask_test]
y_test_6m_clean = y_test_6m[mask_test]


# In[51]:


print(X_train_clean.shape)
print(X_test_clean.shape)
print(y_train_6m_clean.shape)
print(y_test_6m_clean.shape)


# In[52]:


from sklearn.ensemble import RandomForestRegressor

# Create two separate models
#model_3m = RandomForestRegressor(n_estimators=100, random_state=42)
model_6m = RandomForestRegressor(n_estimators=100, random_state=42)

# Train them on their respective targets
#model_3m.fit(X_train_clean, y_train_3m_clean)
model_6m.fit(X_train_clean, y_train_6m_clean)


# In[53]:


from sklearn.metrics import mean_squared_error, r2_score

y_pred_6m = model_6m.predict(X_test_clean)


# In[54]:


print("6-Month Forecast RMSE:", mean_squared_error(y_test_6m_clean, y_pred_6m, squared=False))


# In[55]:


print("6-Month R²:", r2_score(y_test_6m_clean, y_pred_6m))


# In[ ]:





# In[56]:


with open('model_3m.pkl', 'wb') as file:
    pickle.dump(model_3m, file)


# In[57]:


with open('model_6m.pkl', 'wb') as file:
    pickle.dump(model_6m, file)


# In[ ]:





# In[60]:


X_sample = X_test_clean.sample(5)  # or X_test if you want newer data
X_sample.to_csv("sample_input.csv", index=False)


# In[ ]:





# In[ ]:




