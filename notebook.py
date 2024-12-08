#!/usr/bin/env python
# coding: utf-8

# # Student Performance Predictor

# ## Project Domain

# ### Background Problem
# Education is a cornerstone of human development, yet challenges like low academic performance and high dropout rates persist. Factors such as socioeconomic background, study habits, and school environment significantly influence student outcomes. Early identification of at-risk students can enable timely interventions to improve their academic success.
# 
# The UCI Student Performance Dataset provides a rich set of features related to student demographics, behavioral patterns, and academic history, offering an excellent opportunity to apply predictive analytics.
# 
# ### Why and How Should This Problem Be Solved?
# Addressing academic performance issues is crucial because:
# 
# 1. Improving Outcomes: Identifying at-risk students early can help schools and educators implement targeted strategies to improve grades and reduce dropout rates.
# 2. Personalized Interventions: Predictive models can guide personalized support strategies, ensuring better resource allocation.
# 3. Data-Driven Insights: Analyzing key factors affecting performance enables informed decision-making in the education sector.
# 
# 

# ## Business Understanding

# ## Data Understanding

# ## Exploratory Data Analysis

# ### Import All Libraries

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ### Data Loading

# In[2]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# data (as pandas dataframes) 
df = pd.concat([student_performance.data.features, student_performance.data.targets], axis=1)

# metadata 
print(student_performance.metadata) 


# In[3]:


# variable information 
print(student_performance.variables) 


# In[4]:


# Load datasets
math_df = pd.read_csv("student/student-mat.csv", sep=";")
por_df = pd.read_csv("student/student-por.csv", sep=";")

# Define merge columns (key columns)
merge_columns = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", 
                 "Mjob", "Fjob", "reason", "nursery", "internet"]

# Merge the datasets
merge_df = pd.merge(por_df, math_df,  on=merge_columns, suffixes=('_math', '_por'))

# Identify the unique columns
# Keep key columns and the non-key columns from the math_df dataset (with '_math' suffix)
unique_columns = merge_columns + [col for col in merge_df.columns if col.endswith('_math')]

# Rename columns to remove '_math' suffix
merge_df = merge_df[unique_columns]
merge_df.columns = [col.replace('_math', '') for col in merge_df.columns]

# Print the resulting DataFrame's shape
merge_df


# In[5]:


# basic information
merge_df.info()


# ### Initial Inspection

# In[6]:


print("Dataset Information: ")
print(df.info())


# In[7]:


print("Descriptive Statistics:")
print(df.describe())


# In[8]:


print("Missing Values:")
print(df.isnull().sum())


# ### Univariate Analysis

# #### Categorical Columns

# In[9]:


categorical_columns = df.select_dtypes(include=['object', 'category']).columns


# In[10]:


plt.figure(figsize=(20, 15))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(4, 5, i)
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# #### Numerical Columns

# In[11]:


numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns


# In[12]:


plt.figure(figsize=(20,15))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(5, 6, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# ### Multivariate Analysis

# In[13]:


# Categorical vs G3
# Prepare the plot grid
num_columns = 3
num_rows = (len(categorical_columns) + num_columns - 1) // num_columns

plt.figure(figsize=(20, 5 * num_rows))

# Iterate through categorical columns
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(num_rows, num_columns, i)
    
    # Calculate mean G3 for each category
    category_g3_means = df.groupby(col)['G3'].mean().sort_values(ascending=False)
    
    # Create a bar plot
    category_g3_means.plot(kind='bar')
    plt.title(f'Mean Final Grade by {col}')
    plt.xlabel(col)
    plt.ylabel('Mean G3 Grade')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

plt.tight_layout()
plt.show()

# Statistical summary of G3 for each categorical variable
print("Statistical Summary of Final Grade by Categorical Features:")
for col in categorical_columns:
    print(f"\n{col} Categories:")
    print(df.groupby(col)['G3'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False))


# In[14]:


# Correlation Heatmap for Numerical Features
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(20,15))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()


# In[15]:


# Pair Plot for Key Numerical Features
sns.pairplot(df[numerical_columns], diag_kind='kde')
plt.suptitle('Pair Plot of Key Features', y=1.02)
plt.show()


# ## Data Preparation

# ### Encoding Category Features

# In[16]:


categorical_columns


# In[17]:


# Encode categorical columns
for col in categorical_columns:
    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)

df.drop(categorical_columns, axis=1, inplace=True)
# Display the result
print(df.head())


# ### Train Test Split

# In[20]:


from sklearn.model_selection import train_test_split

X = df.drop(['G3'], axis=1)
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


# In[22]:


print(f'Total of sample in whole dataset: {len(X)}')
print(f'Total of sample in train dataset: {len(X_train)}')
print(f'Total of sample in test dataset: {len(X_test)}')


# ### Standardization

# In[23]:


numerical_columns


# In[28]:


numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
scaler = StandardScaler()
scaler.fit(X_train[numerical_columns])

# Transform the training and test data
X_train[numerical_columns] = scaler.transform(X_train.loc[:, numerical_columns])

X_train.head()


# In[31]:


X_train[numerical_columns].describe().round(4)


# ## Modeling

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# In[ ]:


from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)


# In[67]:


# Prepare dataframe for model analysis
models = pd.DataFrame(index=['train_mse', 'test_mse', 'train_mae', 'test_mae', 'train_r2', 'test_r2', 'train_rmse', 'test_rmse'],
                      columns=['Linear_Regression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'])


# ### Linear Regression

# In[69]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

models.loc['train_mse', 'LinearRegression'] = mean_squared_error(y_pred=lin_reg.predict(X_train), y_true=y_train)
models.loc['train_mae', 'LinearRegression'] = mean_absolute_error(y_pred=lin_reg.predict(X_train), y_true=y_train)
models.loc['train_r2', 'LinearRegression'] = r2_score(y_pred=lin_reg.predict(X_train), y_true=y_train)
models.loc['train_rmse', 'LinearRegression'] = np.sqrt(mean_squared_error(y_pred=lin_reg.predict(X_train), y_true=y_train))


# ### K-Nearest Neighbor

# In[70]:


# Hyperparameter tuning
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance']
}

knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_

models.loc['train_mse','KNN'] = mean_squared_error(y_pred = best_knn.predict(X_train), y_true=y_train)
models.loc['train_mse', 'KNN'] = mean_squared_error(y_pred = best_knn.predict(X_train), y_true=y_train)
models.loc['train_mae', 'KNN'] = mean_absolute_error(y_pred = best_knn.predict(X_train), y_true=y_train)
models.loc['train_r2', 'KNN'] = r2_score(y_pred = best_knn.predict(X_train), y_true=y_train)
models.loc['train_rmse', 'KNN'] = np.sqrt(mean_squared_error(y_pred = best_knn.predict(X_train), y_true=y_train))


# ### Random Forest

# In[71]:


# Hyperparameter tuning
param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=best_rf.predict(X_train), y_true=y_train)
models.loc['train_mse', 'RandomForest'] = mean_squared_error(y_pred=best_rf.predict(X_train), y_true=y_train)
models.loc['train_mae', 'RandomForest'] = mean_absolute_error(y_pred=best_rf.predict(X_train), y_true=y_train)
models.loc['train_r2', 'RandomForest'] = r2_score(y_pred=best_rf.predict(X_train), y_true=y_train)
models.loc['train_rmse', 'RandomForest'] = np.sqrt(mean_squared_error(y_pred=best_rf.predict(X_train), y_true=y_train))


# ### XGBoost

# In[ ]:


# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7]
}

xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

models.loc['train_mse','XGBoost'] = mean_squared_error(y_pred=best_xgb.predict(X_train), y_true=y_train)
models.loc['train_mse', 'XGBoost'] = mean_squared_error(y_pred=best_xgb.predict(X_train), y_true=y_train)
models.loc['train_mae', 'XGBoost'] = mean_absolute_error(y_pred=best_xgb.predict(X_train), y_true=y_train)
models.loc['train_r2', 'XGBoost'] = r2_score(y_pred=best_xgb.predict(X_train), y_true=y_train)
models.loc['train_rmse', 'XGBoost'] = np.sqrt(mean_squared_error(y_pred=best_xgb.predict(X_train), y_true=y_train))


# ### Cat Boost

# In[72]:


# Hyperparameter tuning
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'depth': [4, 6, 8]
}

cat = CatBoostRegressor(random_seed=42, verbose=0)
grid_search = GridSearchCV(cat, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_cat = grid_search.best_estimator_

models.loc['train_mse','CatBoost'] = mean_squared_error(y_pred=best_cat.predict(X_train), y_true=y_train)
models.loc['train_mse', 'CatBoost'] = mean_squared_error(y_pred=best_cat.predict(X_train), y_true=y_train)
models.loc['train_mae', 'CatBoost'] = mean_absolute_error(y_pred=best_cat.predict(X_train), y_true=y_train)
models.loc['train_r2', 'CatBoost'] = r2_score(y_pred=best_cat.predict(X_train), y_true=y_train)
models.loc['train_rmse', 'CatBoost'] = np.sqrt(mean_squared_error(y_pred=best_cat.predict(X_train), y_true=y_train))


# ## Evaluation

# In[60]:


X_test.loc[:, numerical_columns] = scaler.transform(X_test[numerical_columns])


# In[61]:


mse = pd.DataFrame(columns=['train', 'test'], index=['Linear_Regression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'])

model_dict = {'Linear_Regression': lin_reg, 'KNN': best_knn, 'RandomForest': best_rf, 'XGBoost': best_xgb, 'CatBoost': best_cat}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 

mse


# In[ ]:


mae = pd.DataFrame(columns=['train', 'test'], index=['Linear_Regression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'])

model_dict = {'Linear_Regression': lin_reg, 'KNN': best_knn, 'RandomForest': best_rf, 'XGBoost': best_xgb, 'CatBoost': best_cat}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 

mse


# In[62]:


fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)


# In[66]:


prediksi = X_test.iloc[:20].copy()
pred_dict = {'y_true':y_test[:20]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)

