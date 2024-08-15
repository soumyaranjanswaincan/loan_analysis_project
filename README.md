## Introduction
In this project, we developed and evaluated three different machine learning models—Logistic Regression, Random Forest, and Neural Networks—to predict the status of loans (either 'Fully Paid' or 'Charged Off'). The goal is to identify which model performs best in classifying loans and to understand the strengths and weaknesses of each approach.

## Data overview 

- Dataset: The dataset consists of various financial and demographic features of loan applicants.
- arget Variable: loan_status, which has two classes:
1 for 'Fully Paid'
0 for 'Charged Off'
- Features: We preprocessed the data by:
1. Dropping irrelevant or redundant columns.
2. Encoding categorical variables using one-hot encoding.
3. Standardizing the data to ensure all features contribute equally to the models.

## Modeling Approaches

1. Logistic Regression
- Purpose: A linear model to estimate the probability of a loan being 'Fully Paid'.
- Class Weight Adjustment: We used class_weight="balanced" to handle class imbalance.

2. Random Forest
- Purpose: An ensemble learning method that builds multiple decision trees and merges them to obtain more accurate and stable predictions.
- Class Weight Adjustment: Similar to Logistic Regression, we adjusted for class imbalance.

3. Neural Network
- Purpose: A more complex model that attempts to capture non-linear relationships between features and the target variable.
- Architecture: A simple feedforward neural network with one hidden layer.

4. Decision Tree
- Decision Trees work well with large datasets and are relatively fast to train
- Decision Trees are easy to interpret and understand
- Can handle non-linear relationships between features and the target variable

5. XGBoost
- High performance and efficiency in handling large datasets.
- Produces accurate predictions
- Incorporates regularization techniques to prevent overfitting.

## Model Evaluation Metrics

- Accuracy: The overall correctness of the model's predictions.
- Precision: The proportion of true positive predictions (e.g., correctly predicted 'Charged Off' loans) to all positive predictions.
- Recall: The ability of the model to capture all true positive cases.
- F1-Score: The harmonic mean of precision and recall, providing a single metric to balance both.

## Results
1. Logistic Regression
Accuracy: 80.35%
Precision (Class 0): 0.55
Recall (Class 0): 0.07
F1-Score (Class 0): 0.12
Precision (Class 1): 0.81
Recall (Class 1): 0.99
F1-Score (Class 1): 0.89

2. Random Forest
Accuracy: 80.29%
Precision (Class 0): 0.53
Recall (Class 0): 0.08
F1-Score (Class 0): 0.13
Precision (Class 1): 0.81
Recall (Class 1): 0.98
F1-Score (Class 1): 0.89

3. Neural Network
Accuracy: 80.26%
Precision (Class 0): 0.52
Recall (Class 0): 0.10
F1-Score (Class 0): 0.17
Precision (Class 1): 0.81
Recall (Class 1): 0.98
F1-Score (Class 1): 0.89

4. Decision Tree
Accuracy: 71.01%
Precision (Class 0): 0.27
Recall (Class 0): 0.28
F1-Score (Class 0): 0.28
Precision (Class 1): 0.82
Recall (Class 1): 0.82
F1-Score (Class 1): 0.82

5. XGBoost
Accuracy: 80.388%
Precision (Class 0): 0.53
Recall (Class 0): 0.10
F1-Score (Class 0): 0.17
Precision (Class 1): 0.81
Recall (Class 1): 0.98
F1-Score (Class 1): 0.89

## Analysis
Overall Performance: Decision Tree has the worest accuracy, around 70% and all other models achieved similar accuracy, around 80%, indicating that they effectively predict the majority class ('Fully Paid'). 
Class Imbalance Challenge: All models struggled with identifying 'Charged Off' loans, resulting in low recall and F1-scores for Class 0.
Neural Network Performance: The Neural Network slightly outperformed the other models in detecting 'Charged Off' loans, showing a small improvement in recall for Class 0. However, this improvement was marginal and did not significantly enhance overall model performance.
Conclusion & Recommendations
Class Imbalance Handling: All models demonstrate difficulty in handling class imbalance, particularly in predicting the minority class ('Charged Off'). We used "class-weight" - balanced and did not find a better result in almost all the models. Logistic Regiression Modul shows even worse result when the class-weight is set to be balanced. Further techniques, such as SMOTE (Synthetic Minority Over-sampling Technique) or using more complex ensemble methods, could be explored to improve performance.
Model Selection: While the Neural Network offered a slight edge, the choice of model should consider the trade-offs between interpretability (Logistic Regression) and performance (Neural Networks). Random Forest provides a good balance with moderate interpretability and competitive performance.
Next Steps: To improve the model's ability to predict 'Charged Off' loans, it would be valuable to explore more sophisticated models or techniques, including ensemble methods, advanced neural network architectures, and feature engineering.

