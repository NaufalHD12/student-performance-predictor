#!/usr/bin/env python
# coding: utf-8

# # Student Performance Predictor - Naufal Hadi Darmawan

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
# ### Research Findings
# - OECD Report on Education (2021) highlights that socio-economic and behavioral factors play significant roles in academic performance.  
# Link: https://www.oecd.org/en/publications/2021/09/education-at-a-glance-2021_dd45f55e.html
# 
# - The regression study explored how Socio-economic predicted academic outcomes while adjusting for other factors like family engagement and school resources. Correlation analysis examined the relationship between SES and academic achievement. The Socio-economic position appears to affect academic performance. Higher-Socio-economic students fare better academically. However, parental participation and school resources may buffer the SES-academic achievement association.
# Link: https://doi.org/10.54183/jssr.v3i2.308

# ## Business Understanding

# ### Problem Statements
# 1. How can we predict a student’s final grade (G3) based on available demographic, behavioral, and academic features?
# 2. What are the most influential factors affecting the final grade, and how can schools use this information to improve academic outcomes?
# 
# ### Goals
# - **Primary Goal**: Build a regression model to predict students’ final grades (G3) accurately.
# - **Secondary Goal**: Identify key predictors of student performance and analyze their relative importance.
# 
# ### Solution Statements
# To achieve the goals, the following approaches will be used:
# 
# 1. Baseline Model:  
# - Use simple algorithms like Linear Regression or Decision Tree Regressor to set a baseline for prediction accuracy.
# - Evaluate using metrics such as MSE, MAE, R², and RMSE.
# 
# 2. Advanced Model:  
# - Implement advanced regression algorithms like Random Forest, Gradient Boosted Trees (XGBoost and CatBoost).
# - Perform hyperparameter tuning (e.g., tree depth, learning rate) to optimize performance.

# ## Data Understanding

# ### Dataset Overview
# The UCI Student Performance dataset includes information on students from two subjects (Math and Portuguese). For this project, merged data will be used.  
# - Number of Rows: 649 (merged dataset).
# - Number of Columns: 33 (including the target G3).
# 
# Link to dataset: https://archive.ics.uci.edu/dataset/320/student+performance
# ### Data Condition
# The UCI Student Performance dataset is clean, with no missing values or outliers, as confirmed by the dataset's documentation on the UCI website. All features are well-defined, and their ranges are appropriate for the context of the data. For example, numeric features like age, absences, and grades fall within logical and expected ranges, while categorical features such as school, sex, and address contain only valid predefined categories. This ensures that the dataset is ready for analysis and modeling without requiring additional steps like imputing missing data or handling anomalies.
# ### Features
# 1. school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)  
# 2. sex - student's sex (binary: 'F' - female or 'M' - male)  
# 3. age - student's age (numeric: from 15 to 22)  
# 4. address - student's home address type (binary: 'U' - urban or 'R' - rural)  
# 5. famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)  
# 6. Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)  
# 7. Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)  
# 8. Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)  
# 9. Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')  
# 10. Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')  
# 11. reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')  
# 12. guardian - student's guardian (nominal: 'mother', 'father' or 'other')  
# 13. traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)  
# 14. studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)  
# 15. failures - number of past class failures (numeric: n if 1<=n<3, else 4)  
# 16. schoolsup - extra educational support (binary: yes or no)  
# 17. famsup - family educational support (binary: yes or no)  
# 18. paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)  
# 19. activities - extra-curricular activities (binary: yes or no)  
# 20. nursery - attended nursery school (binary: yes or no)  
# 21. higher - wants to take higher education (binary: yes or no)  
# 22. internet - Internet access at home (binary: yes or no)  
# 23. romantic - with a romantic relationship (binary: yes or no)  
# 24. famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)  
# 25. freetime - free time after school (numeric: from 1 - very low to 5 - very high)  
# 26. goout - going out with friends (numeric: from 1 - very low to 5 - very high)  
# 27. Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)  
# 28. Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)  
# 29. health - current health status (numeric: from 1 - very bad to 5 - very good)  
# 30. absences - number of school absences (numeric: from 0 to 93)  
# 31. G1 - first period grade (numeric: from 0 to 20)  
# 32. G2 - second period grade (numeric: from 0 to 20)  
# 33. G3 - final grade (numeric: from 0 to 20, **output target**)

# ## Exploratory Data Analysis

# ### Import All Libraries

# In[1]:


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


# ### Initial Inspection
# In this section we perform initial inspection to find out general information (**info()**) and descriptive statistics(**describe()**) on the dataset

# In[4]:


print("Dataset Information: ")
print(df.info())


# In[5]:


print("Descriptive Statistics:")
print(df.describe())


# In[6]:


print("Missing Values:")
print(df.isnull().sum())


# ### Univariate Analysis

# The next step is to perform Univariate Analysis on Categorical Features and Numerical Features to see their distribution

# #### Categorical Columns

# In[7]:


categorical_columns = df.select_dtypes(include=['object', 'category']).columns


# In[8]:


plt.figure(figsize=(20, 15))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(4, 5, i)
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# - **Findings on Categorical Features**:  
#     - **School**: Shows the count of individuals in two different schools, with Gabriel Pereira (GP) having a larger representation.
#     - **Sex**: Distribution of gender, with females slightly outnumbering males.
#     - **Address**: Urban (U) addresses dominate over rural (R).
#     - **Family Size (famsize)**: More families have greater than three members (GT3) than less or equal to three (LE3).
#     - **Parental Status (Pstatus)**: Most individuals have parents living together (T) compared to apart (A).
#     - **Mother's Job (Mjob)**: Varied occupations, with "other" being most common, followed by "services," "at home," and others.
#     - **Father's Job (Fjob)**: Similar pattern to mother's job, with "other" being the highest category.
#     - **Reason for School Choice**: Includes "course" as the most common reason, followed by "home" and others.
#     - **Guardian**: Mothers are the most common guardians, followed by fathers and others.
#     - **School Support (schoolsup)**: Most individuals do not receive school support.
#     - **Family Support (famsup)**: Family support is more commonly present than absent.
#     - **Paid Classes (paid)**: The majority do not attend paid classes.
#     - **Extracurricular Activities (activities)**: Participation is almost evenly split.
#     - Nursery Attended (nursery): Most individuals attended nursery.
#     - **Higher Education Aspirations (higher)**: Most aspire to pursue higher education.
#     - **Internet Access (internet)**: Internet access is more prevalent.
#     - **Romantic Relationships (romantic)**: Fewer individuals are in romantic relationships compared to those who are not.

# #### Numerical Columns

# In[9]:


numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns


# In[10]:


plt.figure(figsize=(20,15))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(5, 6, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# - **Findings on Numerical Features**:  
#     - **Age**: Majority of individuals are between 15 and 18 years old, with a peak at 16-17.
#     - **Mother's Education (Medu)**: Education levels cluster around the midrange, with higher counts for 3 and 4.
#     - **Father's Education (Fedu)**: Similar distribution to Medu, with peaks at 3 and 4.
#     - **Travel Time (traveltime)**: Most individuals have short travel times (category 1), with few in categories 3 and 4.
#     - **Study Time (studytime)**: Predominantly low (1 and 2), with fewer students dedicating more study hours (3 and 4).
#     - **Failures**: Most individuals have zero failures, with progressively fewer as the count increases.
#     - **Family Relationship Quality (famrel)**: Majority report high quality (values 4 and 5).
#     - **Free Time (freetime)**: Most individuals rate their free time between 3 and 4.
#     - **Going Out with Friends (goout)**: Peaks around the middle range (3), with fewer at extremes (1 and 5).
#     - **Daily Alcohol Consumption (Dalc)**: Majority report low consumption (1), with very few reporting high levels (4 or 5).
#     - **Weekly Alcohol Consumption (Walc)**: Slightly more varied than Dalc, but low levels (1 and 2) are most common.
#     - **Health**: Most rate their health as good to very good (4 and 5).
#     - **Absences**: The distribution is right-skewed, with most students having few or no absences.
#     - **Grades (G1, G2, G3)**: Representing academic performance:
#         - Grades (G1, G2, G3) follow a similar pattern, resembling a normal distribution centered around 10-15.

# ### Multivariate Analysis

# The objective of this section is to perform a detailed multivariate analysis of both categorical and numerical features to uncover their relationships with students' final grades (G3). By visualizing statistical summaries, correlation heatmaps, and extracting key insights, this analysis aims to identify the most influential factors

# In[11]:


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


# - Academic performance is influenced by several factors, such as school type, address, parental occupation, and support systems.
# - Students with higher aspirations (e.g., pursuing higher education) and certain advantages (e.g., urban living, educated parents) tend to perform better.
# - Social and environmental factors like extracurricular activities, internet access, and lack of romantic relationships also positively influence grades.

# In[12]:


# Correlation Heatmap for Numerical Features
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(20,15))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()


# - **Summary of Correlation Heatmap**:  
#     - **Key Positive Correlations**:
#         - G1, G2, and G3: Strong positive correlations among these features, indicating a high consistency in students' grades across the three periods (e.g., G1 ↔ G2 = 0.86, G2 ↔ G3 = 0.92, G1 ↔ G3 = 0.83).
#         - Parental Education (Medu, Fedu): Positive correlation between parents' education levels and final grades (G3), with Medu ↔ G3 = 0.24 and Fedu ↔ G3 = 0.21.
#         - Study Time: Slight positive correlation with final grades (studytime ↔ G3 = 0.25).
#     - **Key Negative Correlations**:
#         - Failures: Strong negative correlation with grades, showing that higher failures reduce performance (failures ↔ G3 = -0.39).
#         - Absences: Weak negative correlation with grades (absences ↔ G3 = -0.09).
#         - Alcohol Consumption:
#             - Weekday (Dalc): Weak negative correlation with grades (Dalc ↔ G3 = -0.20).
#             - Weekend (Walc): Similar weak negative correlation (Walc ↔ G3 = -0.18).
#     - **Insights**:
#         - Failures have the most substantial negative impact on grades, while parental education and study time positively influence performance.
#         - Social and lifestyle factors like alcohol consumption, absences, and leisure activities (e.g., going out) show weaker but notable negative associations with academic performance.
#         - Students with consistent grades across G1, G2, and G3 are likely maintaining stable performance, as seen in their high correlations.

# In[13]:


# Pair Plot for Key Numerical Features
sns.pairplot(df[numerical_columns], diag_kind='kde')
plt.suptitle('Pair Plot of Key Features', y=1.02)
plt.show()


# ## Data Preparation

# ### Encoding Category Features

# - **Objective**: Convert categorical columns into a numeric format suitable for machine learning algorithms.
# - **Approach**:
#     - Used one-hot encoding to transform categorical variables into binary columns using **pd.get_dummies**.
#     - Added the encoded columns back to the dataset while dropping the original categorical columns.

# In[14]:


categorical_columns


# In[15]:


# Encode categorical columns
for col in categorical_columns:
    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)

df.drop(categorical_columns, axis=1, inplace=True)
# Display the result
print(df.head())


# ### Train Test Split

# - **Objective**: Split the dataset into training and testing sets for model evaluation.
# - **Approach**:
#     - Separated the features (X) and the target variable (y) where G3 (Final Grade) is the target.
#     - Used train_test_split from sklearn to create training and testing sets.
#     - Allocated 20% of the data for testing and ensured reproducibility by setting a random seed (random_state=123).

# In[16]:


from sklearn.model_selection import train_test_split

X = df.drop(['G3'], axis=1)
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


# In[17]:


print(f'Total of sample in whole dataset: {len(X)}')
print(f'Total of sample in train dataset: {len(X_train)}')
print(f'Total of sample in test dataset: {len(X_test)}')


# ### Standardization

# - **Objective**: Normalize the numerical features to improve model performance and ensure uniform scaling.
# - **Approach**:
#     - Selected relevant numerical columns such as age, family relationships, study time, alcohol consumption, etc.
#     - Used **StandardScaler** from sklearn to scale the training data.
#     - Applied the fitted scaler to the testing data for consistency.

# In[18]:


numerical_columns


# In[19]:


numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
scaler = StandardScaler()
scaler.fit(X_train[numerical_columns])

# Transform the training and test data
X_train[numerical_columns] = scaler.transform(X_train.loc[:, numerical_columns])

X_train.head()


# In[20]:


X_train[numerical_columns].describe().round(4)


# ## Modeling

# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# In[22]:


from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
)


# ### Preparing the DataFrame for Model Analysis
# Create a structured DataFrame to store the evaluation metrics for multiple machine learning models.

# In[23]:


# Prepare dataframe for model analysis
models = pd.DataFrame(index=['train_mse', 'test_mse', 'train_mae', 'test_mae', 'train_r2', 'test_r2', 'train_rmse', 'test_rmse'],
                      columns=['Linear_Regression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'])


# ### Linear Regression

# Linear Regression assumes a linear relationship between independent variables (features) and the dependent variable (target). It minimizes the sum of squared residuals (differences between observed and predicted values) to find the best-fit line.

# In[24]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

models.loc['train_mse', 'LinearRegression'] = mean_squared_error(y_pred=lin_reg.predict(X_train), y_true=y_train)
models.loc['train_mae', 'LinearRegression'] = mean_absolute_error(y_pred=lin_reg.predict(X_train), y_true=y_train)
models.loc['train_r2', 'LinearRegression'] = r2_score(y_pred=lin_reg.predict(X_train), y_true=y_train)
models.loc['train_rmse', 'LinearRegression'] = np.sqrt(mean_squared_error(y_pred=lin_reg.predict(X_train), y_true=y_train))


# ### K-Nearest Neighbor

# KNN Regression predicts the target for a given data point by averaging the targets of its n_neighbors closest points. The closeness is measured using a distance metric (e.g., Euclidean distance).
# - **Hyperparameters Tuned via GridSearchCV**:
#     - The **param_grid** defines a systematic range of hyperparameters to test. For KNN, n_neighbors (number of neighbors) and weights (uniform or distance-based) significantly influence the model's ability to generalize. This ensures that the model's performance is thoroughly explored within a controlled range of options.
#     - **n_neighbors**: Number of neighbors to consider. I use [3, 5, 7, 9, 11].
#         - Effect: A smaller n_neighbors (e.g., 2) makes predictions more sensitive to local variations, which may lead to overfitting. A larger value (e.g., 15) smooths predictions but risks underfitting.
#     - **weights**: Weighting function (uniform or distance).
#         - Effect: uniform gives all neighbors equal importance, while distance assigns greater weight to closer neighbors, improving predictions in datasets where proximity correlates strongly with the target variable.

# In[25]:


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

# Random Forest builds multiple decision trees during training and averages their predictions to improve accuracy and control overfitting.
# - **Hyperparameters Tuned via GridSearchCV**:
#     - Using the **param_grid** tuning setup for the Random Forest Regressor ensures optimal performance by systematically exploring a range of key parameters that significantly impact the model's ability to generalize
#     - **n_estimators**: Number of trees in the forest (e.g., 100, 200, 300).
#         - Effect: Increasing n_estimators improves accuracy by reducing variance, but it increases training time. Diminishing returns are observed after a certain point.
#     - **max_depth**: Maximum depth of each tree (e.g., None, 10, 20).
#         - Effect: A deeper tree can capture more complex patterns but risks overfitting. Shallower trees reduce overfitting but may miss important patterns.
#     - **min_samples_split**: Minimum number of samples required to split a node (e.g., 2, 5, 10).
#         - Effect: Larger values prevent the model from creating overly complex trees, reducing overfitting at the cost of potentially higher bias.

# In[26]:


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

# XGBoost trains sequential trees, where each subsequent tree reduces the errors of the previous ones. It uses regularization techniques to prevent overfitting.
# - **Hyperparameters Tuned via GridSearchCV**:
#     - Using the **param_grid** for the XGBoost Regressor ensures that the model is fine-tuned for optimal performance by systematically exploring key parameters that influence its ability to capture complex relationships and avoid overfitting.
#     - **n_estimators**: Number of boosting rounds (e.g., 100, 200, 300).
#         - Effect: Increasing n_estimators allows the model to learn more from the data, improving accuracy but increasing the risk of overfitting and training time.
#     - **learning_rate**: Shrinkage step size (e.g., 0.01, 0.1, 0.3).
#         - Effect: A smaller learning_rate makes the model learn slowly, requiring more boosting rounds for convergence, while a larger value speeds up learning but risks overshooting the optimal solution.
#     - **max_depth**: Maximum depth of a tree (e.g., 3, 5, 7).
#         - Effect: Deeper trees improve the model's ability to capture complex patterns but increase overfitting risk and training time. Shallower trees may underfit.

# In[27]:


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

# CatBoost improves on gradient boosting by processing categorical features efficiently. It also reduces prediction bias using ordered boosting.
# - **Hyperparameters Tuned via GridSearchCV**:
#     - Use the **param_grid** tuning setup for the CatBoost Regressor is beneficial because it allows you to find the optimal configuration of key hyperparameters that influence model performance, efficiency, and the ability to handle categorical features
#     - **iterations**: Number of boosting rounds (e.g., 100, 200, 300).
#         - Effect: Increasing iterations enhances accuracy but lengthens training time. Too many iterations may cause overfitting.
#     - **learning_rate**: Step size shrinkage (e.g., 0.01, 0.1, 0.3).
#         Effect: A smaller learning_rate ensures better convergence by taking smaller steps but requires more boosting rounds. A larger value speeds up convergence at the risk of overshooting.
#     - **depth**: Depth of the trees (e.g., 4, 6, 8).
#         - Effect: Larger depths allow the model to capture complex patterns but increase overfitting risks, while smaller depths may lead to underfitting.

# In[28]:


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

# In[39]:


X_test[numerical_columns] = scaler.transform(X_test.loc[:, numerical_columns])


# In[40]:


mse = pd.DataFrame(columns=['train', 'test'], index=['Linear_Regression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'])

model_dict = {'Linear_Regression': lin_reg, 'KNN': best_knn, 'RandomForest': best_rf, 'XGBoost': best_xgb, 'CatBoost': best_cat}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 

mse


# In[41]:


mae = pd.DataFrame(columns=['train', 'test'], index=['Linear_Regression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'])

model_dict = {'Linear_Regression': lin_reg, 'KNN': best_knn, 'RandomForest': best_rf, 'XGBoost': best_xgb, 'CatBoost': best_cat}

for name, model in model_dict.items():
    mae.loc[name, 'train'] = mean_absolute_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mae.loc[name, 'test'] = mean_absolute_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 

mae


# In[42]:


r2 = pd.DataFrame(columns=['train', 'test'], index=['Linear_Regression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'])

model_dict = {'Linear_Regression': lin_reg, 'KNN': best_knn, 'RandomForest': best_rf, 'XGBoost': best_xgb, 'CatBoost': best_cat}

for name, model in model_dict.items():
    r2.loc[name, 'train'] = r2_score(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    r2.loc[name, 'test'] = r2_score(y_true=y_test, y_pred=model.predict(X_test))/1e3
 

r2


# In[43]:


rmse = pd.DataFrame(columns=['train', 'test'], index=['Linear_Regression', 'KNN', 'RandomForest', 'XGBoost', 'CatBoost'])

model_dict = {'Linear_Regression': lin_reg, 'KNN': best_knn, 'RandomForest': best_rf, 'XGBoost': best_xgb, 'CatBoost': best_cat}

for name, model in model_dict.items():
    rmse.loc[name, 'train'] = np.sqrt(mean_absolute_error(y_true=y_train, y_pred=model.predict(X_train))/1e3) 
    rmse.loc[name, 'test'] = np.sqrt(mean_absolute_error(y_true=y_test, y_pred=model.predict(X_test))/1e3)
 

rmse


# In[44]:


fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
ax.set_title('Model Comparison: Test MSE', fontsize=16)


# - CatBoost has relatively balanced performance between training and testing sets.
# - Linear Regression has a relatively higher MSE on train set compared to other models
# - Random Forest performs moderately with a small training MSE but higher test MSE.
# - KNN has very high test MSE, which may indicate overfitting or poor generalization.
# - XGBoost has the lowest train MSE but a considerable gap with test MSE, indicating possible overfitting.

# In[45]:


fig, ax = plt.subplots()
mae.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
ax.set_title('Model Comparison: Test MAE', fontsize=16)


# - CatBoost shows good balance with lower MAE in both training and testing datasets.
# - Random Forest has moderate train and test MAE, showing slightly better generalization than other models.
# - Linear Regression maintains consistent but relatively high MAE for both datasets, suggesting underfitting.
# - XGBoost has the smallest train MAE but a notable gap to test MAE, indicating overfitting.
# - KNN displays the highest test MAE, suggesting overfitting or poor performance on unseen data.

# In[46]:


fig, ax = plt.subplots()
r2.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
ax.set_title('Model Comparison: Test R2 Score', fontsize=16)


# - XGBoost demonstrates strong performance with high train and test R², though the train R² is slightly higher, indicating minor overfitting.
# - KNN shows relatively balanced train and test R² but at a lower overall score, suggesting limited model capacity for this problem.
# - Random Forest performs well with a small gap between train and test R², suggesting good generalization.
# - Linear Regression maintains consistent but lower R² scores for both datasets, indicative of underfitting.
# - CatBoost achieves high train and test R² scores, showcasing strong and consistent performance.

# In[47]:


fig, ax = plt.subplots()
rmse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
ax.set_title('Model Comparison: Test RMSE', fontsize=16)


# - CatBoost performs well with a low Test RMSE, suggesting it generalizes well to new data.
# - Random Forest shows a higher Test RMSE, which could indicate some overfitting since the training RMSE is much lower.
# - Linear Regression has a relatively higher RMSE compared to other models, implying less effective performance on this dataset.
# - XGBoost and KNN also show higher Test RMSE compared to CatBoost, with KNN potentially struggling with generalization.

# In[50]:


prediction = X_test.iloc[:20].copy()
pred_dict = {'y_true':y_test[:20]}
for name, model in model_dict.items():
    pred_dict[name + '_prediction'] = model.predict(prediction).round(1)
 
pd.DataFrame(pred_dict)


# ### Final Decision Model
# - Based on the analysis, **CatBoost** emerged as the most suitable model. It consistently achieved:
#     - Low error on the testing set across all metrics (MSE, MAE, RMSE)
#     - High R2 score, indicating it explains a larger proportion of the variance in the new data.
# - This suggests **CatBoost** offers a good balance between training and testing performance, generalizing well to unseen data without overfitting.
