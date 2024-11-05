# Bank Customer Churn Prediction
This project aims to analyze and predict customer churn for a bank using a dataset of 10,000 customers. 
The analysis explores various factors such as demographics, account balance, and product usage, which could influence a customer's likelihood of exiting or churning. 
The process includes detailed data analysis, feature engineering, and model training to achieve meaningful insights and build an effective predictive model.

## Access the Project
[AOL Group Project_Bank Customer Churn Prediction](https://colab.research.google.com/drive/1bIGYUd34rnkq_DKvwre6-lN_kZXs7rBQ?usp=sharing)

## Data Overview
The dataset is loaded from a CSV file, and an initial data overview is performed to understand the structure and features. The dataset contains 14 columns and 10,000 rows, with columns like CreditScore, Geography, Gender, Balance, and Exited indicating whether a customer has churned.

## Exploratory Data Analysis (EDA)
- Data Overview: An overview of top entries reveals key insights into customer demographics and behavior. For example, most customers are in the 35-45 age range, and the majority have a credit card and are active members.
- Checking Data: Duplicates and missing values are checked. No duplicates or missing values were found.
- Feature Distributions: Density plots visualize the distribution of numerical features. Categorical data like Geography, Gender, HasCrCard, and IsActiveMember are analyzed for unique values and counts.
- Churn Breakdown: The data shows 7,963 customers have not churned, while 2,037 have.
    
## Feature Engineering
- Categorization: Columns like Age, CreditScore, Balance, and EstimatedSalary are grouped into ranges for a more categorical view. For instance, Age is split into age groups (e.g., 20-29, 30-39) and visualized against churn status.
- Significant Features: Statistical tests identify significant features influencing churn. Tenure and HasCrCard are dropped due to low significance based on chi-square tests.
  
## Data Preprocessing
- Feature Selection: Non-essential columns such as RowNumber, CustomerId, and Surname are removed.
- Encoding: Categorical variables are encoded for model compatibility.
- Scaling: Numerical columns such as CreditScore, Age, and Balance are scaled using MinMaxScaler to bring all values into a 0-1 range.

## Data Splitting
The dataset is divided into training and test sets using an 80:20 ratio to ensure that the model is trained on a substantial portion of the data while leaving enough for evaluation. Given the dataset's total size of 10,000 records, this results in:
- Training Set: 8,000 records (80%)
- Test Set: 2,000 records (20%)

The training set is used to train and tune the model, while the test set is reserved for assessing model performance on unseen data, which helps evaluate generalization and prevent overfitting. 
This split provides a reliable basis for comparing models and understanding how well they might perform in real-world scenarios.

## Modeling 
To predict customer churn, I used the following machine learning models, each with unique characteristics for handling binary classification tasks:
1. Logistic Regression: This linear model is commonly used as a baseline due to its simplicity and interpretability. It establishes a benchmark for other models, offering insights into the relationships between customer features and churn probability.
2. Random Forest Classifier: This ensemble model constructs multiple decision trees and averages their predictions, reducing overfitting while capturing complex patterns in the data. It works well with both numerical and categorical data, making it effective for this task.
3. Gaussian Naive Bayes: Based on Bayes’ theorem, this model assumes independence among predictors and is particularly efficient with categorical data. Although it simplifies the problem by assuming Gaussian distribution for continuous features, it often performs well in baseline comparisons.
4. LightGBM Classifier: A gradient boosting framework that uses tree-based learning algorithms, LightGBM is optimized for speed and performance. It handles large datasets efficiently and provides high accuracy by iteratively building models that correct previous errors.

## Model Evaluation and Best Model
The models were evaluated using metrics such as Accuracy, Precision, Recall, and F1-Score. 
Additionally, AUC-ROC (Area Under the Receiver Operating Characteristic curve) was used to assess the models' ability to differentiate between the two classes (churn vs. no churn).
Among the tested models, Gradient Boosting (specifically XGBoost or LightGBM) generally performed the best for this dataset. 
These models handled the mixture of categorical and numerical variables effectively and showed strong performance in terms of both accuracy and AUC-ROC scores. 
The fine-tuning of hyperparameters (such as learning rate, number of estimators, and max depth) further improved the model's performance.

## Feature Importance
Key features identified as influential in predicting customer churn include:
- Age: Older customers tended to have a higher likelihood of churning.
- Balance: Higher balances correlated with lower churn rates.
- CreditScore: A lower credit score was often associated with a higher chance of exiting.
Using these important features, the model was fine-tuned to improve accuracy and AUC-ROC scores, providing valuable insights for the bank to target retention efforts effectively.
