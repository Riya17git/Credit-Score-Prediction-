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


# In[2]:


card = pd.read_csv('cards_data.csv')
transaction = pd.read_csv('transactions_data.csv')
user = pd.read_csv('users_data.csv')


# In[3]:


transaction.head(5)


# In[5]:


card.head(3)


# In[6]:


user.head(3)


# In[4]:


transaction = transaction.sample(frac=0.5, random_state=42)


# # Information of Datasets

# In[5]:


card.shape


# In[6]:


transaction.shape


# In[7]:


user.shape


# In[8]:


def basic_info(df):
  print('Shape of the dataset is', df.shape)
  print('-'*50)
  print('information of dataset is', df.info())
  print('-'*50)
  print('Description of dataset is', df.describe())
  print('-'*50)
  print('Null values in dataset is', df.isnull().sum())
  print('-'*50)
  print('Duplicate values in dataset is', df.duplicated().sum())
  print('-'*50)
  print('Unique values in dataset is', df.nunique())
  print('-'*50)
  print('Columns of dataset is', df.columns)
  print('-'*50)
  print('datatype of columns', df.dtypes)
  print('-'*50)


# #  Changing the Datatypes to save memory

# In[9]:


int_cols = ['id','client_id','cvv','num_cards_issued','year_pin_last_changed']
float_cols = ['card_number']

card[int_cols] = card[int_cols].astype('int32')
card[float_cols] = card[float_cols].astype('float32')

print(card.info())


# In[10]:


transaction = transaction.sample(frac=0.5, random_state=42)


# In[11]:


int_cols_user = ['id','current_age','retirement_age','birth_year','credit_score','num_credit_cards','birth_month']
float_cols_user = ['latitude','longitude']

user[int_cols_user] = user[int_cols_user].astype('int32')
user[float_cols_user] = user[float_cols_user].astype('float32')

print(user.info())


#  ## Importing mcc_codes and placing in transaction table

# In[8]:


with open('mcc_codes.json') as f:
  mcc_code = json.load(f)


# In[9]:


with open('mcc_codes.json') as f:
  mcc_code = json.load(f)


# In[10]:


transaction['mcc_description'] = transaction['mcc'].map(mcc_code)


# In[11]:


transaction.head(3)


# #  Merging the Dataset

# In[15]:


merged_df = pd.merge(transaction,card, on='client_id', how = 'inner')
merged_df.head()


# In[16]:


merged_data = pd.merge(merged_df,user, left_on='client_id', right_on='id', how = 'inner')
merged_data.head()


# In[17]:


merged_data.shape


# #  Cleaning the Data

# In[18]:


merged_data = merged_data.rename(columns={'id_x':'id_transaction','id_y':'id_card'})


# In[19]:


merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data['expires'] = pd.to_datetime(merged_data['expires'],format='%m/%Y')
merged_data['acct_open_date'] = pd.to_datetime(merged_data['acct_open_date'],format='%m/%Y')


# In[20]:


merged_data['amount'] = merged_data['amount'].replace('[\$,]', '', regex=True).astype(float)
merged_data['credit_limit'] = merged_data['credit_limit'].replace('[\$,]', '', regex=True).astype(float)
merged_data['per_capita_income'] = merged_data['per_capita_income'].replace('[\$,]', '', regex=True).astype(float)
merged_data['yearly_income'] = merged_data['yearly_income'].replace('[\$,]', '', regex=True).astype(float)
merged_data['total_debt'] = merged_data['total_debt'].replace('[\$,]', '', regex=True).astype(float)


# In[21]:


merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data['year'] = merged_data['date'].dt.year
merged_data['month'] = merged_data['date'].dt.month
merged_data['day'] = merged_data['date'].dt.day


# #  Spilliting the dataset

# In[65]:


x = merged_data.drop('credit_score',axis = 1)
y = merged_data['credit_score']


# In[66]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# #  Handling Null Values

# In[67]:


missing_percentage = (x_train.isnull().sum()/len(x_train) * 100)
print(missing_percentage)


# In[68]:


x_train.shape


# In[69]:


x_train = x_train.drop('errors',axis='columns')
x_train = x_train.drop('date',axis='columns')


# # We are checking before imputing null values that columns come under which category ?
# 
# 1. MCAR (Missing Completely at Random)
# 2. MAR (Missing at Random)
# 3. MCNR (Missing Not At Random)

# #    ## Here 0 means not missing and 1 Means missing.
# ## Both groups have a similar median credit score (~700+).
# ## This suggests missing values in all columns do not strongly impact credit_score.
# ## The spread (interquartile range, IQR) looks almost identical for both groups.
# ## This further supports that missing values are not heavily influencing credit_score.
# ## Since the distributions are very similar, missingness in all columns does not strongly affect credit_score.
# ## Missing values in all columns are likely MAR (Missing at Random) rather than NMAR.

# In[70]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[71]:


x_train.head(3)


# #  Handling categorical variables before imputing 

# #  Label Encoding before filling null values

# In[72]:


label_column = ['merchant_city','merchant_state','mcc_description']
one_hot_column = ['use_chip','card_brand','card_type','has_chip','gender','card_on_dark_web']

label_encoder = LabelEncoder()
for col in label_column:
    x_train[col] = label_encoder.fit_transform(x_train[col])
    
x_train = pd.get_dummies(x_train, columns=one_hot_column)


# In[73]:


x_train = x_train.astype(int, errors='ignore')


# In[74]:


x_train = x_train.drop('address',axis='columns')
x_train = x_train.drop('id_transaction',axis = 'columns')
x_train = x_train.drop('card_number',axis = 'columns')


# In[75]:


x_train.head(2)


# #   Handling null values

# In[76]:


x_train.isnull().sum()


# In[80]:


imputer_median = SimpleImputer(strategy="median")
#x_train['amount'] = imputer_median.fit_transform(x_train[['amount']])


# In[82]:


imputer_mode = SimpleImputer(strategy="most_frequent")
#x_train['merchant_id'] = imputer_mode.fit_transform(x_train[['merchant_id']])


# In[31]:


#x_train['merchant_city'] = imputer_mode.fit_transform(x_train[['merchant_city']])


# In[83]:


x_train['zip'] = imputer_mode.fit_transform(x_train[['zip']])


# In[33]:


#x_train['mcc'] = imputer_median.fit_transform(x_train[['mcc']])


# In[34]:


#x_train[['year', 'month', 'day']] = imputer_mode.fit_transform(x_train[['year', 'month', 'day']])


# In[84]:


x_train.isnull().sum()


# #  Handling Null and categorical values in X_test

# In[85]:


missing_percentage = (x_test.isnull().sum()/len(x_test) * 100)
print(missing_percentage)


# In[86]:


x_test.head(3)


# In[87]:


x_test = x_test.drop('date',axis='columns')
x_test = x_test.drop('errors',axis='columns')
x_test = x_test.drop('address',axis='columns')
x_test = x_test.drop('id_transaction',axis='columns')
x_test = x_test.drop('card_number',axis='columns')


# In[88]:


label_column = ['merchant_city','merchant_state','mcc_description']
one_hot_column = ['use_chip','card_brand','card_type','has_chip','gender','card_on_dark_web']

label_encoder = LabelEncoder()
for col in label_column:
    x_test[col] = label_encoder.fit_transform(x_test[col])
    
x_test = pd.get_dummies(x_test, columns=one_hot_column)


# In[89]:


x_test = x_test.astype(int, errors='ignore')


# In[90]:


x_test.head(3)


# In[91]:


x_test.isnull().sum()


# In[43]:


#x_test = x_test.dropna(subset=['id_transaction'])
#x_test = x_test.dropna(subset=['card_id'])


# In[44]:


#imputer_median = SimpleImputer(strategy="median")
#x_test['amount'] = imputer_median.fit_transform(x_test[['amount']])


# In[92]:


imputer_mode = SimpleImputer(strategy="most_frequent")
#x_test['merchant_id'] = imputer_mode.fit_transform(x_test[['merchant_id']])


# In[46]:


#x_test['merchant_city'] = imputer_mode.fit_transform(x_test[['merchant_city']])


# In[93]:


x_test['zip'] = imputer_mode.fit_transform(x_test[['zip']])


# In[48]:


#x_test['mcc'] = imputer_median.fit_transform(x_test[['mcc']])


# In[49]:


#x_test[['year', 'month', 'day']] = imputer_mode.fit_transform(x_test[['year', 'month', 'day']])


# In[94]:


x_test.isnull().sum()


# # Handling outliers in x_train

# In[53]:


for column in x_train.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(5, 4))  # Set figure size for each plot
    plt.boxplot(x_train[column])  # Plot boxplot
    plt.title(f'Boxplot of {column}')  # Add title
    plt.xlabel(column)  # X-axis label
    plt.show()  


# In[95]:


x_train['amount'].min()


# In[96]:


x_train['amount'].max()


# In[97]:


x_train['amount'].mean()


# In[98]:


plt.figure(figsize=(10, 6))
sns.histplot(x_train['amount'],bins=50)
plt.title(f'amount distribution')
plt.show()


# In[99]:


Q1 = x_train['amount'].quantile(0.25)
Q3 = x_train['amount'].quantile(0.75)
IQR = Q3 - Q1

outliers = x_train[(x_train['amount'] < (Q1 - 1.5 * IQR)) | (x_train['amount'] > (Q3 + 1.5 * IQR))]
#print(outliers.head(10))
print(outliers.shape)


# In[100]:


x_train.shape


# In[68]:


plt.figure(figsize=(8, 4))
sns.histplot(x_train['amount'], bins=50, kde=True)
plt.title('Distribution of Amount')
plt.show()


# In[94]:


col = ['per_capita_income','yearly_income','total_debt']

for i in col:
    plt.figure(figsize=(8, 4))
    sns.histplot(x_train[i], bins=50, kde=True)
    plt.show()
    


# # Applied log transformer on amount column

# In[101]:


x_train['amount'] = np.log1p(x_train['amount']) 


# In[54]:


plt.figure(figsize=(8, 4))
sns.histplot(x_train['amount'], bins=50, kde=True)
plt.title('Distribution of Amount')
plt.show()


# In[102]:


Q1 = x_train['amount'].quantile(0.25)
Q3 = x_train['amount'].quantile(0.75)
IQR = Q3 - Q1

outliers = x_train[(x_train['amount'] < (Q1 - 1.5 * IQR)) | (x_train['amount'] > (Q3 + 1.5 * IQR))]
#print(outliers.head(10))
print(outliers.shape) ## new extreme outliers


# In[103]:


x_train = x_train[~x_train.index.isin(outliers.index)]


# # Removing those rows from y_train as well 

# In[104]:


y_train = y_train.loc[x_train.index]


# In[105]:


print(x_train.shape)


# In[106]:


print(y_train.shape)


# In[107]:


print(x_train['credit_limit'].min())
print(x_train['credit_limit'].max())
print(x_train['credit_limit'].mean())


# In[108]:


Q1 = x_train['credit_limit'].quantile(0.25)
Q3 = x_train['credit_limit'].quantile(0.75)
IQR = Q3 - Q1

outliers = x_train[(x_train['credit_limit'] < (Q1 - 1.5 * IQR)) | (x_train['credit_limit'] > (Q3 + 1.5 * IQR))]
#print(outliers.head(10))
print(outliers.shape)


# In[64]:


plt.figure(figsize=(8, 4))
sns.histplot(x_train['credit_limit'], bins=50, kde=True)
plt.title('Distribution of Amount')
plt.show()


# In[81]:


#x_train_cleaned['credit_limit'] = np.sqrt(x_train_cleaned['credit_limit'])


# # by applying log transform it is even increasing outliers.
# # So I am capping the values.

# In[109]:


x_train['credit_limit'] = winsorize(x_train['credit_limit'], limits=[0.05, 0.05])  # Capping top and bottom 1%


# In[65]:


plt.figure(figsize=(8, 4))
sns.histplot(x_train_cleaned['credit_limit'], bins=50, kde=True)
plt.title('Distribution of Amount')
plt.show()


# In[110]:


Q1 = x_train['credit_limit'].quantile(0.25)
Q3 = x_train['credit_limit'].quantile(0.75)
IQR = Q3 - Q1

outliers = x_train[(x_train['credit_limit'] < (Q1 - 1.5 * IQR)) | (x_train['credit_limit'] > (Q3 + 1.5 * IQR))]
#print(outliers.head(10))
print(outliers.shape)


# In[ ]:





# In[111]:


print(x_train['retirement_age'].min())
print(x_train['retirement_age'].max())
print(x_train['retirement_age'].mean())


# In[67]:


plt.figure(figsize=(8, 4))
sns.histplot(x_train_cleaned['retirement_age'], bins=50, kde=True)
plt.title('Distribution of retirement_age')
plt.show()


# In[68]:


plt.figure(figsize=(8, 4))
sns.histplot(x_train_cleaned['per_capita_income'], bins=50, kde=True)
plt.title('Distribution of per capita income')
plt.show()


# In[112]:


print(x_train['per_capita_income'].min())
print(x_train['per_capita_income'].max())
print(x_train['per_capita_income'].mean())


# #   categorical column outlier distribution 

# In[69]:


import seaborn as sns
import matplotlib.pyplot as plt

for col in x_train.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(y=x_train[col], order=x_train[col].value_counts().index)  # Plot bar chart
    plt.title(f'Distribution of {col}')
    plt.show()


# In[70]:


for col in x_train.select_dtypes(include=['object']).columns:  # Select categorical columns
    print(f"\nColumn: {col}")
    print(x_train[col].value_counts())  # Display category counts


# In[ ]:





# #  Handling outlier in x_test

# In[70]:


for column in x_test.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(5, 4))  # Set figure size for each plot
    plt.boxplot(x_test[column])  # Plot boxplot
    plt.title(f'Boxplot of {column}')  # Add title
    plt.xlabel(column)  # X-axis label
    plt.show()  


# In[113]:


print(x_test['amount'].min())
print(x_test['amount'].max())
print(x_test['amount'].mean())


# In[76]:


plt.figure(figsize=(8, 4))
sns.histplot(x_test['amount'], bins=50, kde=True)
plt.title('Distribution of Amount')
plt.show()


# In[114]:


Q1 = x_test['amount'].quantile(0.25)
Q3 = x_test['amount'].quantile(0.75)
IQR = Q3 - Q1

outliers = x_test[(x_test['amount'] < (Q1 - 1.5 * IQR)) | (x_test['amount'] > (Q3 + 1.5 * IQR))]
#print(outliers.head(10))
print(outliers.shape) 


# In[115]:


x_test['amount'] = np.log1p(x_test['amount']) 


# In[116]:


Q1 = x_test['amount'].quantile(0.25)
Q3 = x_test['amount'].quantile(0.75)
IQR = Q3 - Q1

outliers = x_test[(x_test['amount'] < (Q1 - 1.5 * IQR)) | (x_test['amount'] > (Q3 + 1.5 * IQR))]
#print(outliers.head(10))
print(outliers.shape) ## new extreme outliers


# In[117]:


x_test = x_test[~x_test.index.isin(outliers.index)]


# In[118]:


y_test = y_test.loc[x_test.index]


# In[71]:


plt.figure(figsize=(8, 4))
sns.histplot(x_test['amount'], bins=50, kde=True)
plt.title('Distribution of Amount')
plt.show()


# #  credit_limit column

# In[119]:


Q1 = x_test['credit_limit'].quantile(0.25)
Q3 = x_test['credit_limit'].quantile(0.75)
IQR = Q3 - Q1

outliers = x_test[(x_test['credit_limit'] < (Q1 - 1.5 * IQR)) | (x_test['credit_limit'] > (Q3 + 1.5 * IQR))]
#print(outliers.head(10))
print(outliers.shape) 


# In[120]:


x_test['credit_limit'] = winsorize(x_test['credit_limit'], limits=[0.05, 0.05])  # Capping top and bottom 3%


# In[121]:


Q1 = x_test['credit_limit'].quantile(0.25)
Q3 = x_test['credit_limit'].quantile(0.75)
IQR = Q3 - Q1

outliers = x_test[(x_test['credit_limit'] < (Q1 - 1.5 * IQR)) | (x_test['credit_limit'] > (Q3 + 1.5 * IQR))]
#print(outliers.head(10))
print(outliers.shape) ## new extreme outliers


# In[77]:


plt.figure(figsize=(8, 4))
sns.histplot(x_test['credit_limit'], bins=50, kde=True)
plt.title('Distribution of Credit Limit')
plt.show()


# In[82]:


col = ['per_capita_income','yearly_income','total_debt']

for i in col:
    plt.figure(figsize=(8, 4))
    sns.histplot(x_test[i], bins=50, kde=True)
    plt.show()


# In[122]:


x_train.shape


# In[123]:


x_test.shape


# In[124]:


y_train.shape


# In[125]:


y_test.shape


# In[126]:


y_train.isnull().sum()


# In[127]:


y_test.isnull().sum()


# # After applying log tranform null values are there in amount column

# In[83]:


#imputer_median = SimpleImputer(strategy="median")
#x_train['amount'] = imputer_median.fit_transform(x_train[['amount']])


# In[84]:


#x_test['amount'] = imputer_median.fit_transform(x_test[['amount']])


# In[128]:


x_train.head()


# #  Feature Extraction in x_tarin

# In[129]:


x_train['acct_open_year'] = x_train['acct_open_date'].dt.year


# In[130]:


x_train['acct_open_month'] = x_train['acct_open_date'].dt.month


# In[131]:


x_train['acct_open_date'] = x_train['acct_open_date'].dt.day


# In[132]:


x_train = x_train.drop('acct_open_date',axis='columns')


# #  Handling expires column

# In[133]:


x_train['expires_year'] = x_train['expires'].dt.year
x_train['expires_month'] = x_train['expires'].dt.month
x_train['expires_date'] = x_train['expires'].dt.day


# In[134]:


x_train = x_train.drop('expires',axis='columns')


# In[ ]:





# #  How many years since account is open

# In[135]:


x_train['years_since_acct_open'] = 2025 - x_train['acct_open_year']


# # last year when pin changed

# In[136]:


x_train['years_since_pin_changed'] = 2025 - x_train['year_pin_last_changed']


# #  what is average amount per client id

# In[137]:


x_train['avg_transaction_amount'] = x_train.groupby('client_id')['amount'].transform('mean')


# # Creating credit_utilization

# In[139]:


x_train['credit_utilization'] = x_train['amount'] / x_train['credit_limit']


# In[140]:


x_train.shape


# # Feature Extraction in x_test

# In[141]:


x_test['acct_open_year'] = x_test['acct_open_date'].dt.year
x_test['acct_open_month'] = x_test['acct_open_date'].dt.month
x_test['acct_open_date'] = x_test['acct_open_date'].dt.day


# In[142]:


x_test = x_test.drop('acct_open_date',axis='columns')


# In[143]:


x_test.head(3)


# #   Handling Expires column

# In[144]:


x_test['expires_year'] = x_test['expires'].dt.year
x_test['expires_month'] = x_test['expires'].dt.month
x_test['expires_date'] = x_test['expires'].dt.day


# In[145]:


x_test = x_test.drop('expires',axis='columns')


# In[146]:


x_test.head(3)


# #   How many years since account is open
# 

# In[147]:


x_test['years_since_acct_open'] = 2025 - x_test['acct_open_year']


# # last year when pin changed

# In[148]:


x_test['years_since_pin_changed'] = 2025 - x_test['year_pin_last_changed']


# # what is average amount per client id

# In[149]:


x_test['avg_transaction_amount'] = x_test.groupby('client_id')['amount'].transform('mean')


# # Creating credit_utilization

# In[150]:


x_test['credit_utilization'] = x_test['amount'] / x_test['credit_limit']


# In[151]:


x_test.shape


# In[152]:


x_train.shape


# In[153]:


y_train.shape


# In[154]:


y_test.shape


# In[155]:


x_train.head()


# In[1]:


x_train.info()


# In[157]:


x_test.info()


# #   Feature Scalling

# In[116]:


#x_train_scaled = x_train.drop(['expires','acct_open_date'],axis=1)
#x_test_scaled = x_test.drop(['expires','acct_open_date'],axis=1)


# In[158]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# #  Model Training

# In[159]:


rf_classifier = RandomForestRegressor(n_estimators=100, random_state=42)


# In[160]:


rf_classifier.fit(x_train, y_train)


# In[161]:


y_pred = rf_classifier.predict(x_test)


# In[162]:


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)


# In[ ]:





# # Expense Forcast model

# In[1]:


import pandas as pd
import numpy as np
import datetime
import json


# In[2]:


card = pd.read_csv('cards_data.csv')
transaction = pd.read_csv('transactions_data.csv')
user = pd.read_csv('users_data.csv')


# In[3]:


transaction = transaction.sample(frac=0.5, random_state=42)


# In[4]:


int_cols_transaction = ['id','client_id','card_id','merchant_id','mcc']
float_cols_transaction = ['zip']

transaction[int_cols_transaction] = transaction[int_cols_transaction].astype('int32')
transaction[float_cols_transaction] = transaction[float_cols_transaction].astype('float32')

print(transaction.info())


# In[5]:


with open('mcc_codes.json') as f:
  mcc_code = json.load(f)


# In[6]:


transaction['mcc'] = transaction['mcc'].astype(str)


# In[7]:


transaction['mcc_description'] = transaction['mcc'].map(mcc_code)


# In[ ]:





# In[8]:


transaction.head(3)


# In[9]:


import pandas as pd

# Assuming you have a dataframe named `df`
transaction['date'] = pd.to_datetime(transaction['date'])
transaction['month_year'] = transaction['date'].dt.to_period('M')

# Remove dollar sign and convert to numeric
transaction['amount'] = transaction['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Group by client and month_year, summing the amount
monthly_expenses = transaction.groupby(['client_id', 'month_year'])['amount'].sum().reset_index()


# In[11]:


print(monthly_expenses)


# In[44]:


client_data = monthly_expenses[
    monthly_expenses['client_id'] == 1759
]['amount'].groupby(monthly_expenses['month_year']).sum().values


# In[45]:


actual_last_month = client_data[-1]


# In[46]:


print(f"Last month's actual expense: ${actual_last_month:.2f}")


# In[ ]:





# In[47]:


from sklearn.preprocessing import MinMaxScaler

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
monthly_expenses['normalized_amount'] = scaler.fit_transform(monthly_expenses[['amount']])

# Create a time series for each client
def create_timeseries(client_data, time_steps=3):
    X, y = [], []
    for i in range(len(client_data) - time_steps):
        X.append(client_data[i:i + time_steps])
        y.append(client_data[i + time_steps])
    return np.array(X), np.array(y)

# Example for one client
client_data = monthly_expenses[monthly_expenses['client_id'] == 1759]['normalized_amount'].values
X, y = create_timeseries(client_data)


# In[ ]:





# In[ ]:





# In[48]:


import torch
import torch.nn as nn

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Create the model, loss function, and optimizer
model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[49]:


print("X shape:", X.shape)
print("y shape:", y.shape)


# In[ ]:





# In[50]:


# Convert data to tensors
X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y, dtype=torch.float32)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_function(output, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


# In[ ]:





# In[51]:


# Make predictions for the next month
model.eval()
with torch.no_grad():
    predicted = model(X_train[-1:]).item()

# Reverse normalization
predicted_expense = scaler.inverse_transform([[predicted]])[0][0]
print(f"Predicted expense for next month: ${predicted_expense:.2f}")


# In[64]:


torch.save(model, 'expense_last_model.pkl')


# In[65]:


print(monthly_expenses['month_year'].head())


# #    Creating expenses for multiple categories

# In[52]:


card = pd.read_csv('cards_data.csv')
transaction = pd.read_csv('transactions_data.csv')
user = pd.read_csv('users_data.csv')


# In[53]:


transaction = transaction.sample(frac=0.5, random_state=42)


# In[54]:


int_cols_transaction = ['id','client_id','card_id','merchant_id','mcc']
float_cols_transaction = ['zip']

transaction[int_cols_transaction] = transaction[int_cols_transaction].astype('int32')
transaction[float_cols_transaction] = transaction[float_cols_transaction].astype('float32')

print(transaction.info())


# In[55]:


with open('mcc_codes.json') as f:
  mcc_code = json.load(f)


# In[56]:


transaction['mcc'] = transaction['mcc'].astype(str)


# In[57]:


transaction['mcc_description'] = transaction['mcc'].map(mcc_code)


# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


transaction['date'] = pd.to_datetime(transaction['date'])
transaction['month_year'] = transaction['date'].dt.to_period('M')
transaction['amount'] = transaction['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Group by client, month, and category
monthly_category_expenses = transaction.groupby(
    ['client_id', 'month_year', 'mcc_description']
)['amount'].sum().reset_index()


# In[59]:


target_categories = [
    "Drug Stores and Pharmacies",
    "Grocery Stores, Supermarkets",
    "Utilities - Electric, Gas, Water, Sanitary",
    "Eating Places and Restaurants",
    "Taxicabs and Limousines"
]


# In[60]:


client_id = 1759
time_steps = 3

categories = [
    cat for cat in target_categories
    if cat in monthly_category_expenses[
        monthly_category_expenses['client_id'] == client_id
    ]['mcc_description'].unique()
]

predictions = {}

for category in categories:
    cat_data = monthly_category_expenses[
        (monthly_category_expenses['client_id'] == client_id) &
        (monthly_category_expenses['mcc_description'] == category)
    ]['amount'].fillna(0).values

    if len(cat_data) > time_steps:
        scaled_data = scaler.fit_transform(cat_data.reshape(-1, 1)).flatten()
        X, y = create_timeseries(scaled_data, time_steps=3)
        X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y, dtype=torch.float32)

        # Train model (or use pre-trained one)
        model = LSTMModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = loss_function(output, y_train)
            loss.backward()
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            next_val = model(X_train[-1:]).item()
            predicted_amount = scaler.inverse_transform([[next_val]])[0][0]
            predictions[category] = predicted_amount


# In[61]:


for cat, amount in predictions.items():
    print(f"Predicted next month's spending on {cat}: ${amount:.2f}")


# In[63]:


monthly_category_expenses.to_csv("monthly_expenses.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:




