import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def run_task1():
    df = pd.read_csv("Dataset.csv")
    
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    
    label_encoders = {}
    categorical_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
    features = ['Price range', 'Votes', 'Average Cost for two', 'Has Table booking', 'Has Online delivery']
    target = 'Aggregate rating'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("=" * 60)
    print("TASK 1: RESTAURANT RATING PREDICTION RESULTS")
    print("=" * 60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        
        print(f"\nModel: {name}")
        print(f"Mean Squared Error (MSE) : {mse:.4f}")
        print(f"Root Mean Squared Error  : {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-Squared (R2 Score)     : {r2:.4f}")
        
    rf_model = models["Random Forest Regressor"]
    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importances (Random Forest):")
    print(feature_imp_df.to_string(index=False))
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
    plt.title('Feature Importances for Restaurant Rating Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig("task1_feature_importance.png", dpi=300)
    plt.close()
    
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv("task1_model_evaluation.csv")
    print("\nSaved evaluation summary to task1_model_evaluation.csv and plot to task1_feature_importance.png")

if __name__ == "__main__":
    run_task1()
