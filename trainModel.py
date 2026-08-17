import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv('cleaned_student_academic_risk_dataset.csv')

X = df[
    [
        "Attendance",
        "QuizAverage",
        "AssignmentAverage",
        "MoodleActivity",
        "PreviousAverage"
    ]
]

y = df["Risk Level"]

print(X.head())

# encode target labels
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nRisk Classes")

for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")

# Save encoder
joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

# Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=5,
    stratify=y
)
print("\nTraining samples: ", len(X_train))
print("\nTesting samples: ", len(X_test))

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nReady to train machine learning model")
# Machine Learning model

# store model results
results = {}

best_model = None
best_model_name = ""
best_accuracy = 0

def evaluate_model(model, model_name):

    print("\n==============================================")
    print(f"Training {model_name}")
    print("==============================================")

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # confusion matrix
    confusionMatrix = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy : .4f}")
    print(f"Precision: {precision : .4f}")
    print(f"Recall: {recall : .4f}")
    print(f"F1 Score: {f1 : .4f}")

    print("\nConfusion Matrix")
    print(confusionMatrix)

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    results[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "model": model
    }

# Logistic Regression

logistic_model = LogisticRegression(
    max_iter=100,
    random_state=5
)


evaluate_model(
    logistic_model,
    "Logistic Regression"
)

# Decision Tree

decision_tree_model = DecisionTreeClassifier(
    random_state=5
)

evaluate_model(
    decision_tree_model,
    "Decision Tree"
)

# Random Forest

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=5
)

evaluate_model(
    random_forest,
    "Random Forest"
)

# Find best model

print("\n ----------------------------------------------------------")
print("Model Comparison")
print("------------------------------------------------------------")

for model_name, values in results.items():
    print(
        f"{model_name:25}"
        f"Accuracy: {values['accuracy']: .4f}"
    )

    if values['accuracy'] > best_accuracy:
        best_accuracy = values['accuracy']
        best_model_name = model_name
        best_model = values['model']

print("\n ----------------------------------------------------------")
print(f"Best Model: {best_model_name}")
print(f"Accuracy : {best_accuracy:.4f}")
print("------------------------------------------------------------")

print("\nSaving best model...")

joblib.dump(
    best_model,
    "models/model.pkl"
)

print("Model saved successfully.")
print("Location: models/model.pkl")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

import matplotlib.pyplot as plt

# Only tree-based models have feature importance
if hasattr(best_model, "feature_importances_"):

    feature_names = X.columns

    importance = best_model.feature_importances_

    feature_importance = pd.DataFrame({

        "Feature": feature_names,
        "Importance": importance

    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance")
    print(feature_importance)

    plt.figure(figsize=(10,6))

    plt.bar(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.title("Feature Importance")

    plt.xlabel("Features")

    plt.ylabel("Importance")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig("models/feature_importance.png")



    print("Feature importance graph saved.")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

from sklearn.metrics import ConfusionMatrixDisplay

predictions = best_model.predict(X_test)

disp = ConfusionMatrixDisplay.from_predictions(

    y_test,

    predictions,

    display_labels=label_encoder.classes_,

    cmap="Blues"

)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("models/confusion_matrix.png")



print("Confusion matrix saved.")

# ==========================================================
# SAVE MODEL INFORMATION
# ==========================================================

summary = pd.DataFrame([
    {
        "Model": model_name,
        "Accuracy": values["accuracy"],
        "Precision": values["precision"],
        "Recall": values["recall"],
        "F1 Score": values["f1"],
        "Best Model": model_name == best_model_name
    }
    for model_name, values in results.items()
])

summary.to_csv(

    "models/model_summary.csv",

    index=False

)

print("Model summary saved.")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n==============================================")
print("TRAINING COMPLETE")
print("==============================================")

print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_accuracy:.4f}")

print("\nFiles Generated")

print("----------------------------------------------")

print("models/model.pkl")

print("models/scaler.pkl")

print("models/label_encoder.pkl")

print("models/confusion_matrix.png")

print("models/feature_importance.png")

print("models/model_summary.csv")

print("----------------------------------------------")

print("Ready for prediction.")