# Task 3: Cuisine Classification Model

## 1. Project Overview & Objective
The objective of **Task 3** is to design and train a supervised Machine Learning classification model to predict the primary cuisine of a restaurant based on operating features like pricing, average cost, popular vote counts, aggregate ratings, and service capabilities (table booking and online delivery).

## 2. Target Variable Formulation & Data Processing
Restaurants in the dataset frequently offer multiple multi-cuisine combinations listed in comma-separated strings (e.g., `"North Indian, Chinese, Fast Food"`).
- **Primary Cuisine Extraction**: The first listed cuisine term is identified as the primary culinary domain.
- **Class Grouping**: To handle long-tail rare cuisines and prevent extreme class imbalance, the top 8 most frequent primary cuisines (e.g., North Indian, Chinese, Fast Food, South Indian, Cafe, Italian, etc.) are kept as explicit target categories, while remaining infrequent categories are grouped into an `'Other'` label.

## 3. Selected Feature Matrix
- **`Price range`**: Numerical price index (1 to 4).
- **`Votes`**: Customer popularity and volume metric.
- **`Average Cost for two`**: Monetary metric.
- **`Aggregate rating`**: Overall quality score.
- **`Has Table booking`**: Encoded binary flag (0 or 1).
- **`Has Online delivery`**: Encoded binary flag (0 or 1).

## 4. Model Training & Evaluation Setup
- **Algorithm**: **Random Forest Classifier** (`n_estimators=100`, `random_state=42`).
- **Data Splitting**: Stratified 80-20 Train-Test split to ensure proportionate class representation across train and test partitions.
- **Evaluation Metrics**: Overall Accuracy, Class-wise Precision, Recall, Macro/Weighted F1-Scores, and Confusion Matrix heatmap visualization.

## 5. Key Findings & Insights
- **Pricing & Cost Separation**: High-end cuisines (e.g., Fine Dining Italian / Mughlai) display distinct price range distributions compared to Fast Food or Street Food outlets.
- **Delivery Preferences**: Fast Food and Chinese outlets exhibit higher online delivery proportions compared to cafes.
- **Confusion Matrix Analysis**: Minor misclassifications occur between closely overlapping dining formats (e.g., Cafe vs Fast Food), but major culinary categories maintain distinct feature boundaries.

## 6. Deliverables & Output Files
- **Python Execution Script**: [task3_cuisine_classification.py](file:///C:/Users/Admin/.gemini/antigravity/scratch/cognifyz_ml_internship/task3_cuisine_classification.py)
- **Classification Performance Metrics CSV**: `task3_classification_metrics.csv`
- **Confusion Matrix Heatmap**: `task3_confusion_matrix.png`
