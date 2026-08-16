# Task 1: Restaurant Rating Prediction

## 1. Project Overview & Objective
The primary objective of **Task 1** is to develop and evaluate Machine Learning regression models to predict the aggregate rating (`Aggregate rating`) of a restaurant based on various features such as price range, vote count, average cost for two, table booking availability, and online delivery availability.

## 2. Dataset Overview & Selected Features
The model utilizes key quantitative and categorical features extracted from the dataset:
- **`Price range`**: Categorical indicator of price level (1 = Cheap to 4 = Expensive).
- **`Votes`**: Total number of user reviews/votes received by the restaurant.
- **`Average Cost for two`**: Average monetary cost for two people dining.
- **`Has Table booking`**: Binary indicator (Yes/No) of table reservation service.
- **`Has Online delivery`**: Binary indicator (Yes/No) of food delivery availability.
- **`Aggregate rating`**: Target continuous variable ranging from 0.0 to 5.0.

## 3. Data Preprocessing Pipeline
1. **Handling Missing Values**: Categorical features like `Cuisines` were imputed with `'Unknown'` to avoid missing entries.
2. **Categorical Encoding**: Binary categorical variables (`Has Table booking`, `Has Online delivery`) were converted to numerical representations using `LabelEncoder`.
3. **Train-Test Split**: The dataset was partitioned into an **80% Training set** and a **20% Testing set** using a fixed random state (`random_state=42`) for reproducibility.

## 4. Machine Learning Algorithms
Three distinct regression algorithms were trained and compared:
1. **Linear Regression**: Baseline parametric model establishing linear relationship baseline.
2. **Decision Tree Regressor**: Non-linear tree model capturing non-linear feature splits.
3. **Random Forest Regressor**: Ensemble bagging model of 100 decision trees to reduce variance and improve generalization.

## 5. Model Evaluation Metrics
Models were evaluated using four standard regression metrics:
- **Mean Squared Error (MSE)**: Measures average squared difference between actual and predicted ratings.
- **Root Mean Squared Error (RMSE)**: Square root of MSE, representing error on the original rating scale (0 to 5).
- **Mean Absolute Error (MAE)**: Average magnitude of prediction errors.
- **$R^2$ Score (Coefficient of Determination)**: Proportion of rating variance explained by the model features.

## 6. Feature Importance & Interpretability
Based on the Random Forest Regressor analysis:
- **`Votes`** emerged as the single most influential feature predicting aggregate rating. Higher vote counts strongly correlate with established, highly rated restaurants.
- **`Price range`** and **`Average Cost for two`** also significantly impact the rating, showing that pricing structure plays a vital role in customer perception.
- **`Has Table booking`** and **`Has Online delivery`** provide secondary boosts to rating expectations.

## 7. Deliverables & Output Files
- **Python Execution Script**: [task1_predict_restaurant_ratings.py](file:///C:/Users/Admin/.gemini/antigravity/scratch/cognifyz_ml_internship/task1_predict_restaurant_ratings.py)
- **Evaluation Metrics Summary CSV**: `task1_model_evaluation.csv`
- **Feature Importance Plot**: `task1_feature_importance.png`
