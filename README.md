# Machine Learning Project Report - Naufal Hadi Darmawan ([GitHub](https://github.com/NaufalHD12))
# Student Performance Predictor

## Project Domain
### Background Problem
Education is a cornerstone of human development, yet challenges like low academic performance and high dropout rates persist. Factors such as socioeconomic background, study habits, and school environment significantly influence student outcomes. Early identification of at-risk students can enable timely interventions to improve their academic success.

The UCI Student Performance Dataset provides a rich set of features related to student demographics, behavioral patterns, and academic history, offering an excellent opportunity to apply predictive analytics.

### Why and How Should This Problem Be Solved?
Addressing academic performance issues is crucial because:

- Improving Outcomes: Identifying at-risk students early can help schools and educators implement targeted strategies to improve grades and reduce dropout rates.
- Personalized Interventions: Predictive models can guide personalized support strategies, ensuring better resource allocation.
- Data-Driven Insights: Analyzing key factors affecting performance enables informed decision-making in the education sector.

### Research Findings
- OECD Report on Education (2021) highlights that socio-economic and behavioral factors play significant roles in academic performance.  
Link: https://www.oecd.org/en/publications/2021/09/education-at-a-glance-2021_dd45f55e.html

- The regression study explored how Socio-economic predicted academic outcomes while adjusting for other factors like family engagement and school resources.
Link: https://doi.org/10.54183/jssr.v3i2.308

## Business Understanding
### Problem Statements
- How can we predict a student’s final grade (G3) based on available demographic, behavioral, and academic features?
- What are the most influential factors affecting the final grade, and how can schools use this information to improve academic outcomes?

### Goals
- **Primary Goal**: Build a regression model to predict students’ final grades (G3) accurately.
- **Secondary Goal**: Identify key predictors of student performance and analyze their relative importance.

### Solution Statements
To achieve the goals, the following approaches will be used:

- **Baseline Model**:  
    - Use simple algorithms like Linear Regression to set a baseline for prediction accuracy.
    - Evaluate using metrics such as MSE, MAE, R², and RMSE.

- **Advanced Model**:  
    - Implement advanced regression algorithms like Random Forest, Gradient Boosted Trees (XGBoost and CatBoost).
    - Perform hyperparameter tuning (e.g., tree depth, learning rate) to optimize performance.

## Data Understanding
### Dataset Overview
The UCI Student Performance dataset includes information on students from two subjects (Math and Portuguese). For this project, merged data will be used.  
- Number of Rows: 649 (merged dataset).
- Number of Columns: 33 (including the target G3).

Link to dataset: https://archive.ics.uci.edu/dataset/320/student+performance

### Data Condition
The UCI Student Performance dataset is clean, with no missing values or outliers, as confirmed by the dataset's documentation on the UCI website. All features are well-defined, and their ranges are appropriate for the context of the data. For example, numeric features like age, absences, and grades fall within logical and expected ranges, while categorical features such as school, sex, and address contain only valid predefined categories. This ensures that the dataset is ready for analysis and modeling without requiring additional steps like imputing missing data or handling anomalies.
### Features
1. school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)  
2. sex - student's sex (binary: 'F' - female or 'M' - male)  
3. age - student's age (numeric: from 15 to 22)  
4. address - student's home address type (binary: 'U' - urban or 'R' - rural)  
5. famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)  
6. Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)  
7. Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)  
8. Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)  
9. Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')  
10. Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')  
11. reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')  
12. guardian - student's guardian (nominal: 'mother', 'father' or 'other')  
13. traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)  
14. studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)  
15. failures - number of past class failures (numeric: n if 1<=n<3, else 4)  
16. schoolsup - extra educational support (binary: yes or no)  
17. famsup - family educational support (binary: yes or no)  
18. paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)  
19. activities - extra-curricular activities (binary: yes or no)  
20. nursery - attended nursery school (binary: yes or no)  
21. higher - wants to take higher education (binary: yes or no)  
22. internet - Internet access at home (binary: yes or no)  
23. romantic - with a romantic relationship (binary: yes or no)  
24. famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)  
25. freetime - free time after school (numeric: from 1 - very low to 5 - very high)  
26. goout - going out with friends (numeric: from 1 - very low to 5 - very high)  
27. Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)  
28. Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)  
29. health - current health status (numeric: from 1 - very bad to 5 - very good)  
30. absences - number of school absences (numeric: from 0 to 93)  
31. G1 - first period grade (numeric: from 0 to 20)  
32. G2 - second period grade (numeric: from 0 to 20)  
33. G3 - final grade (numeric: from 0 to 20, **output target**)
### Exploratory Data Analysis
- Loading Data from ucimlrepo
- Perform initial inspection to find out general information (**info()**) and descriptive statistics(**describe()**) on the dataset. 
- Perform Univariate Analysis on Categorical Features and Numerical Features to see their distribution.  
    - **Findings on Categorical Features**:  
    ![univariate categorical](./Image%20Visualization/univariate%20categorical.png)  
        - **School**: Shows the count of individuals in two different schools, with Gabriel Pereira (GP) having a larger representation.
        - **Sex**: Distribution of gender, with females slightly outnumbering males.
        - **Address**: Urban (U) addresses dominate over rural (R).
        - **Family Size (famsize)**: More families have greater than three members (GT3) than less or equal to three (LE3).
        - **Parental Status (Pstatus)**: Most individuals have parents living together (T) compared to apart (A).
        - **Mother's Job (Mjob)**: Varied occupations, with "other" being most common, followed by "services," "at home," and others.
        - **Father's Job (Fjob)**: Similar pattern to mother's job, with "other" being the highest category.
        - **Reason for School Choice**: Includes "course" as the most common reason, followed by "home" and others.
        - **Guardian**: Mothers are the most common guardians, followed by fathers and others.
        - **School Support (schoolsup)**: Most individuals do not receive school support.
        - **Family Support (famsup)**: Family support is more commonly present than absent.
        - **Paid Classes (paid)**: The majority do not attend paid classes.
        - **Extracurricular Activities (activities)**: Participation is almost evenly split.
        - Nursery Attended (nursery): Most individuals attended nursery.
        - **Higher Education Aspirations (higher)**: Most aspire to pursue higher education.
        - **Internet Access (internet)**: Internet access is more prevalent.
        - **Romantic Relationships (romantic)**: Fewer individuals are in romantic relationships compared to those who are not.

    - **Findings on Numerical Features**:  
    ![univariate numerical](./Image%20Visualization/univariate%20numerical.png)
        - **Age**: Majority of individuals are between 15 and 18 years old, with a peak at 16-17.
        - **Mother's Education (Medu)**: Education levels cluster around the midrange, with higher counts for 3 and 4.
        - **Father's Education (Fedu)**: Similar distribution to Medu, with peaks at 3 and 4.
        - **Travel Time (traveltime)**: Most individuals have short travel times (category 1), with few in categories 3 and 4.
        - **Study Time (studytime)**: Predominantly low (1 and 2), with fewer students dedicating more study hours (3 and 4).
        - **Failures**: Most individuals have zero failures, with progressively fewer as the count increases.
        - **Family Relationship Quality (famrel)**: Majority report high quality (values 4 and 5).
        - **Free Time (freetime)**: Most individuals rate their free time between 3 and 4.
        - **Going Out with Friends (goout)**: Peaks around the middle range (3), with fewer at extremes (1 and 5).
        - **Daily Alcohol Consumption (Dalc)**: Majority report low consumption (1), with very few reporting high levels (4 or 5).
        - **Weekly Alcohol Consumption (Walc)**: Slightly more varied than Dalc, but low levels (1 and 2) are most common.
        - **Health**: Most rate their health as good to very good (4 and 5).
        - **Absences**: The distribution is right-skewed, with most students having few or no absences.
        - **Grades (G1, G2, G3)**: Representing academic performance:
            - Grades (G1, G2, G3) follow a similar pattern, resembling a normal distribution centered around 10-15.

- Perform Multivariate Analysis on Categorical Features and Numerical Features  
![Final Grade by Categorical Features](./Image%20Visualization/Final%20Grade%20by%20Categorical%20Features.png)
    - **Insights on Statistical Summary of Final Grade by Categorical Features**: 
        - Academic performance is influenced by several factors, such as school type, address, parental occupation, and support systems.
        - Students with higher aspirations (e.g., pursuing higher education) and certain advantages (e.g., urban living, educated parents) tend to perform better.
        - Social and environmental factors like extracurricular activities, internet access, and lack of romantic relationships also positively influence grades.

    - **Summary of Correlation Heatmap**:  
![Corr Heatmap](./Image%20Visualization/Correlation%20Heatmap.png)
        - **Key Positive Correlations**:
            - G1, G2, and G3: Strong positive correlations among these features, indicating a high consistency in students' grades across the three periods (e.g., G1 ↔ G2 = 0.86, G2 ↔ G3 = 0.92, G1 ↔ G3 = 0.83).
            - Parental Education (Medu, Fedu): Positive correlation between parents' education levels and final grades (G3), with Medu ↔ G3 = 0.24 and Fedu ↔ G3 = 0.21.
            - Study Time: Slight positive correlation with final grades (studytime ↔ G3 = 0.25).
        - **Key Negative Correlations**:
            - Failures: Strong negative correlation with grades, showing that higher failures reduce performance (failures ↔ G3 = -0.39).
            - Absences: Weak negative correlation with grades (absences ↔ G3 = -0.09).
            - Alcohol Consumption:
                - Weekday (Dalc): Weak negative correlation with grades (Dalc ↔ G3 = -0.20).
                - Weekend (Walc): Similar weak negative correlation (Walc ↔ G3 = -0.18).
        - **Insights**:
            - Failures have the most substantial negative impact on grades, while parental education and study time positively influence performance.
            - Social and lifestyle factors like alcohol consumption, absences, and leisure activities (e.g., going out) show weaker but notable negative associations with academic performance.
            - Students with consistent grades across G1, G2, and G3 are likely maintaining stable performance, as seen in their high correlations.

## Data Preparation
### Encoding Categorical Features  
- **Objective**: Convert categorical columns into a numeric format suitable for machine learning algorithms.
- **Approach**:
    - Used one-hot encoding to transform categorical variables into binary columns using **pd.get_dummies**.
    - Added the encoded columns back to the dataset while dropping the original categorical columns.
### Splitting the Dataset
- **Objective**: Split the dataset into training and testing sets for model evaluation.
- **Approach**:
    - Separated the features (X) and the target variable (y) where G3 (Final Grade) is the target.
    - Used train_test_split from sklearn to create training and testing sets.
    - Allocated 20% of the data for testing and ensured reproducibility by setting a random seed (random_state=123).
### Standardizing Numerical Features
- **Objective**: Normalize the numerical features to improve model performance and ensure uniform scaling.
- **Approach**:
    - Selected relevant numerical columns such as age, family relationships, study time, alcohol consumption, etc.
    - Used **StandardScaler** from sklearn to scale the training data.
    - Applied the fitted scaler to the testing data for consistency.

## Modeling
### Preparing the DataFrame for Model Analysis
- **Objective**: Create a structured DataFrame to store the evaluation metrics for multiple machine learning models.
- **Approach**:  
Initialized a DataFrame with rows representing evaluation metrics (MSE, MAE, R², and RMSE) and columns representing the models (Linear Regression, KNN, Random Forest, XGBoost, and CatBoost).
### Linear Regression
- **Objective**: Use a simple regression model as a baseline.
- **Approach**:
    - **Working Mechanism**: Linear Regression assumes a linear relationship between independent variables (features) and the dependent variable (target). It minimizes the sum of squared residuals (differences between observed and predicted values) to find the best-fit line.
    - Trained the LinearRegression model using sklearn.
    - Predicted on the training set and calculated the following metrics:
        - Mean Squared Error (MSE): Measures average squared error.
        - Mean Absolute Error (MAE): Measures average absolute error.
        - R² Score: Indicates the proportion of variance explained.
        - Root Mean Squared Error (RMSE): Square root of MSE.
### K-Nearest Neighbors (KNN) Regression
- **Objective**: Explore a non-parametric method based on proximity between data points.
- **Approach**:
    - **Working Mechanism**: KNN Regression predicts the target for a given data point by averaging the targets of its n_neighbors closest points. The closeness is measured using a distance metric (e.g., Euclidean distance).
    - **Hyperparameters Tuned via GridSearchCV**:
        - The **param_grid** defines a systematic range of hyperparameters to test. For KNN, n_neighbors (number of neighbors) and weights (uniform or distance-based) significantly influence the model's ability to generalize. This ensures that the model's performance is thoroughly explored within a controlled range of options.
        - **n_neighbors**: Number of neighbors to consider. I use [3, 5, 7, 9, 11].
            - Effect: A smaller n_neighbors (e.g., 2) makes predictions more sensitive to local variations, which may lead to overfitting. A larger value (e.g., 15) smooths predictions but risks underfitting.
        - **weights**: Weighting function (uniform or distance).
            - Effect: uniform gives all neighbors equal importance, while distance assigns greater weight to closer neighbors, improving predictions in datasets where proximity correlates strongly with the target variable.
    - Evaluated the best model on training data using the same metrics as Linear Regression.
### Random Forest Regression
- **Objective**: Use a robust ensemble model that handles non-linear relationships.
- **Approach**:
    - **Working Mechanism**: Random Forest builds multiple decision trees during training and averages their predictions to improve accuracy and control overfitting.
    - **Hyperparameters Tuned via GridSearchCV**:
        - Using the **param_grid** tuning setup for the Random Forest Regressor ensures optimal performance by systematically exploring a range of key parameters that significantly impact the model's ability to generalize
        - **n_estimators**: Number of trees in the forest (e.g., 100, 200, 300).
            - Effect: Increasing n_estimators improves accuracy by reducing variance, but it increases training time. Diminishing returns are observed after a certain point.
        - **max_depth**: Maximum depth of each tree (e.g., None, 10, 20).
            - Effect: A deeper tree can capture more complex patterns but risks overfitting. Shallower trees reduce overfitting but may miss important patterns.
        - **min_samples_split**: Minimum number of samples required to split a node (e.g., 2, 5, 10).
            - Effect: Larger values prevent the model from creating overly complex trees, reducing overfitting at the cost of potentially higher bias.
    - Evaluated the best model using the same metrics as other models.
### XGBoost Regression
- **Objective**: Train a gradient-boosted decision tree model for better performance on structured data.
- **Approach**:
    - **Working Mechanism**: XGBoost trains sequential trees, where each subsequent tree reduces the errors of the previous ones. It uses regularization techniques to prevent overfitting.
    - **Hyperparameters Tuned via GridSearchCV**:
        - Using the **param_grid** for the XGBoost Regressor ensures that the model is fine-tuned for optimal performance by systematically exploring key parameters that influence its ability to capture complex relationships and avoid overfitting.
        - **n_estimators**: Number of boosting rounds (e.g., 100, 200, 300).
            - Effect: Increasing n_estimators allows the model to learn more from the data, improving accuracy but increasing the risk of overfitting and training time.
        - **learning_rate**: Shrinkage step size (e.g., 0.01, 0.1, 0.3).
            - Effect: A smaller learning_rate makes the model learn slowly, requiring more boosting rounds for convergence, while a larger value speeds up learning but risks overshooting the optimal solution.
        - **max_depth**: Maximum depth of a tree (e.g., 3, 5, 7).
            - Effect: Deeper trees improve the model's ability to capture complex patterns but increase overfitting risk and training time. Shallower trees may underfit.
    - Evaluated the best model using the same metrics as other models.
### CatBoost Regression
- **Objective**: Train a gradient-boosting model optimized for categorical features and high-speed performance.
- **Approach**:
    - **Working Mechanism**: CatBoost improves on gradient boosting by processing categorical features efficiently. It also reduces prediction bias using ordered boosting.
    - **Hyperparameters Tuned via GridSearchCV**:
        - Use the **param_grid** tuning setup for the CatBoost Regressor is beneficial because it allows you to find the optimal configuration of key hyperparameters that influence model performance, efficiency, and the ability to handle categorical features
        - **iterations**: Number of boosting rounds (e.g., 100, 200, 300).
            - Effect: Increasing iterations enhances accuracy but lengthens training time. Too many iterations may cause overfitting.
        - **learning_rate**: Step size shrinkage (e.g., 0.01, 0.1, 0.3).
            Effect: A smaller learning_rate ensures better convergence by taking smaller steps but requires more boosting rounds. A larger value speeds up convergence at the risk of overshooting.
        - **depth**: Depth of the trees (e.g., 4, 6, 8).
            - Effect: Larger depths allow the model to capture complex patterns but increase overfitting risks, while smaller depths may lead to underfitting.
    - Evaluated the best model using the same metrics as other models.

## Evaluation
###  Metrics that I use:  
- **Mean Squared Error (MSE)**
    - MSE measures the average squared difference between the predicted values and the actual values.
    - A small MSE indicates that the predicted values are close to the actual values.  
- **Mean Absolute Error (MAE)**
    - MAE measures the average of the absolute differences between predicted and actual values.
    - A smaller MAE means better model predictions.  
- **R² Score (Coefficient of Determination)**
    - R² measures the proportion of variance in the dependent variable that is predictable from the independent variables.
    - A higher R² score indicates better model performance. However, it doesn’t consider overfitting, so it should be used alongside other metrics.  
- **Root Mean Squared Error (RMSE)**
    - RMSE is the square root of MSE, representing the average prediction error in the same units as the target variable.
    - A lower RMSE indicates better performance, and the value is directly comparable to the scale of the target variable.
### Why Use These Metrics Together?
Using multiple metrics gives a comprehensive evaluation of my model:
- MSE highlights large errors and ensures robust predictions.
- MAE provides an intuitive, easy-to-interpret average error measure.
- R² offers insight into the model’s explanatory power and variance coverage.
- RMSE bridges the gap between interpretability and penalizing large deviations.

By combining these metrics:
- I ensure the model is evaluated from different perspectives (e.g., sensitivity to outliers, interpretability, and variance explanation).
- Avoid relying on a single metric, which might not reflect all aspects of the model’s performance.
### Result
#### **Model Comparison MSE**  
![Model Comparison Test MSE](./Image%20Visualization/Model%20Comparison%20Test%20MSE.png)  
- CatBoost has relatively balanced performance between training and testing sets.
- Linear Regression has a relatively higher MSE on train set compared to other models
- Random Forest performs moderately with a small training MSE but higher test MSE.
- KNN has very high test MSE, which may indicate overfitting or poor generalization.
- XGBoost has the lowest train MSE but a considerable gap with test MSE, indicating possible overfitting.
#### **Model Comparison MAE**  
![Model Comparison Test MAE](./Image%20Visualization/Model%20Comparison%20Test%20MAE.png)  
- CatBoost shows good balance with lower MAE in both training and testing datasets.
- Random Forest has moderate train and test MAE, showing slightly better generalization than other models.
- Linear Regression maintains consistent but relatively high MAE for both datasets, suggesting underfitting.
- XGBoost has the smallest train MAE but a notable gap to test MAE, indicating overfitting.
- KNN displays the highest test MAE, suggesting overfitting or poor performance on unseen data.
#### **Model Comparison R² Score**  
![Model Comparison Test R2 Score](./Image%20Visualization/Model%20Comparison%20Test%20R2%20Score.png)  
- XGBoost demonstrates strong performance with high train and test R², though the train R² is slightly higher, indicating minor overfitting.
- KNN shows relatively balanced train and test R² but at a lower overall score, suggesting limited model capacity for this problem.
- Random Forest performs well with a small gap between train and test R², suggesting good generalization.
- Linear Regression maintains consistent but lower R² scores for both datasets, indicative of underfitting.
- CatBoost achieves high train and test R² scores, showcasing strong and consistent performance.
#### **Model Comparison RMSE**
![Model Comparison Test RMSE](./Image%20Visualization/Model%20Comparison%20Test%20RMSE.png)  
- CatBoost performs well with a low Test RMSE, suggesting it generalizes well to new data.
- Random Forest shows a higher Test RMSE, which could indicate some overfitting since the training RMSE is much lower.
- Linear Regression has a relatively higher RMSE compared to other models, implying less effective performance on this dataset.
- XGBoost and KNN also show higher Test RMSE compared to CatBoost, with KNN potentially struggling with generalization.

### Prediction Testing
This table presents a comparison of actual target values (y_true) against predictions made by various regression models on a sample of test data. The predictions are rounded to one decimal place for better readability.

![Prediction Testing](./Image%20Visualization/Predict%20Test.png)

**Table Explanation:**
- y_true: The actual values of the target variable (final grades, G3).
- Linear_Regression_prediction: Predictions made by the Linear Regression model.
- KNN_prediction: Predictions from the K-Nearest Neighbors (KNN) model.
- RandomForest_prediction: Predictions from the Random Forest model.
- XGBoost_prediction: Predictions from the XGBoost model.
- CatBoost_prediction: Predictions from the CatBoost model.

### Final Decision Model
- Based on the analysis, **CatBoost** emerged as the most suitable model. It consistently achieved:
    - Low error on the testing set across all metrics (MSE, MAE, RMSE)
    - High R2 score 85% on the testing set, indicating it explains a larger proportion of the variance in the new data.
- This suggests **CatBoost** offers a good balance between training and testing performance, generalizing well to unseen data without overfitting.

## Closing
In this project, I successfully built a machine learning pipeline to predict student academic performance using the UCI Student Performance Dataset. By combining exploratory data analysis, feature engineering, and rigorous modeling techniques, I evaluated multiple regression algorithms, including Linear Regression, KNN, Random Forest, XGBoost, and CatBoost.

Among the tested models, CatBoost demonstrated superior performance, achieving the lowest error metrics (MSE, MAE, RMSE) and the highest R² score. This indicates that CatBoost effectively captures complex relationships in the data while generalizing well to unseen samples.

The analysis revealed several key factors influencing academic performance, such as parental education, study time, past failures, and behavioral patterns like alcohol consumption and social activities. These insights can help educators and policymakers design targeted interventions to improve student outcomes.
