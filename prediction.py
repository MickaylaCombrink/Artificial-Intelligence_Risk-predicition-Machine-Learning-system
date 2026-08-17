import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os


def main():
    # Ensure plots folder exists
    os.makedirs("plots", exist_ok=True)

    # Load dataset
    df = pd.read_csv("student_academic_risk_dataset_100.csv")

    # Basic info
    print(df.info())
    print(df.describe().T)
    print(df.head())
    print(df.isnull().sum())

    # Clean data
    df.drop_duplicates(inplace=True)
    df.drop("StudentID", axis="columns", inplace=True)
    print(df.info())

    # Encode risk levels
    print(df["Risk"].unique().tolist())
    risk_mapping = {"Low": 0, "Moderate": 1, "High": 2}
    risk_level = {v: k for k, v in risk_mapping.items()}
    df["Risk Level"] = df["Risk"].map(risk_mapping)
    df.drop("Risk", axis=1, inplace=True)
    print(df.head())

    # Inputs/outputs
    all_vars = df.columns
    output_var_name = "Risk Level"
    input_var_names = all_vars.drop(output_var_name).to_list()
    df_inputs = df[input_var_names]
    df_output = df[output_var_name]
    print(f"There are {len(input_var_names)} Input Variables")
    print(df_inputs.head())

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(ax=ax)
    ax.set_title("Box Plot: Raw Data")
    fig.savefig("plots/boxplot.png", dpi=300)

    # Histograms
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(df.columns):
        plt.subplot(4, 4, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.savefig("plots/histograms.png", dpi=300)

    # Export cleaned dataset
    df.to_csv("cleaned_student_academic_risk_dataset.csv", index=False)
    print("Cleaned dataset exported to 'cleaned_student_academic_risk_dataset.csv'")
    print("Graphs exported to the 'plots/' folder.")


if __name__ == "__main__":
    main()
