import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def run_task3():
    df = pd.read_csv("Dataset.csv")
    
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    df['Primary_Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())
    
    top_cuisines = df['Primary_Cuisine'].value_counts().head(8).index.tolist()
    df['Cuisine_Target'] = df['Primary_Cuisine'].apply(lambda x: x if x in top_cuisines else 'Other')
    
    label_encoders = {}
    binary_cols = ['Has Table booking', 'Has Online delivery']
    for col in binary_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
    features = ['Price range', 'Votes', 'Average Cost for two', 'Aggregate rating', 'Has Table booking', 'Has Online delivery']
    X = df[features]
    y = df['Cuisine_Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print("=" * 60)
    print("TASK 3: CUISINE CLASSIFICATION MODEL RESULTS")
    print("=" * 60)
    print(f"\nOverall Model Accuracy: {acc:.4f}\n")
    print("Detailed Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix - Cuisine Classification')
    plt.xlabel('Predicted Cuisine')
    plt.ylabel('Actual Cuisine')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("task3_confusion_matrix.png", dpi=300)
    plt.close()
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv("task3_classification_metrics.csv")
    print("Saved confusion matrix plot to task3_confusion_matrix.png and metrics to task3_classification_metrics.csv")

if __name__ == "__main__":
    run_task3()
