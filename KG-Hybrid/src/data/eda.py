"""
File: eda.py

Purpose:
Perform Exploratory Data Analysis (EDA) on the Crop Recommendation dataset.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without opening a GUI

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

def load_dataset():
    """Load the crop recommendation dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    return pd.read_csv(DATASET_PATH)


# ---------------------------------------------------
# Class Distribution
# ---------------------------------------------------

def class_distribution(df):
    """Plot crop class distribution."""

    plt.figure(figsize=(12, 6))

    sns.countplot(
        data=df,
        x="label",
        order=df["label"].value_counts().index
    )

    plt.title("Crop Class Distribution")
    plt.xlabel("Crop")
    plt.ylabel("Number of Samples")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------
# Feature Histograms
# ---------------------------------------------------

def feature_histograms(df):
    """Plot histograms for all numerical features."""

    numerical_features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    for feature in numerical_features:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            bins=20,
            kde=True
        )

        plt.title(f"{feature} Distribution")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{feature}_histogram.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()
# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------

def correlation_heatmap(df):
    """Generate and save the correlation heatmap."""

    numerical_df = df.drop(columns=["label"])

    correlation_matrix = numerical_df.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ---------------------------------------------------
# Box Plots
# ---------------------------------------------------

def feature_boxplots(df):
    """Generate boxplots for numerical features."""

    numerical_features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    for feature in numerical_features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=feature
        )

        plt.title(f"{feature} Boxplot")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{feature}_boxplot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()
# ---------------------------------------------------
# Pair Plot
# ---------------------------------------------------

def pair_plot(df):
    """
    Generate pair plot for selected features.
    """

    selected_features = [
        "N",
        "P",
        "K",
        "temperature",
        "label"
    ]

    pair_df = df[selected_features]

    sns.pairplot(
        pair_df,
        hue="label",
        diag_kind="hist",
        corner=True
    )

    plt.savefig(
        OUTPUT_DIR / "pairplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ---------------------------------------------------
# Main Function
# ---------------------------------------------------

def main():

    df = load_dataset()

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nDataset Shape : {df.shape}")

    print(f"\nNumber of Crops : {df['label'].nunique()}")

    print("\nCrop Names:\n")
    print(sorted(df["label"].unique()))

    print("\nGenerating Class Distribution...")

    class_distribution(df)

    print("Generating Feature Histograms...")

    feature_histograms(df)

    print("Generating Correlation Heatmap...")

    correlation_heatmap(df)

    print("\nEDA Completed Successfully!")

    print(f"\nFigures saved in:\n{OUTPUT_DIR}")
    print("Generating Correlation Heatmap...")
    correlation_heatmap(df)

    print("Generating Boxplots...")
    feature_boxplots(df)
    print("Generating Pair Plot...")
    pair_plot(df)


if __name__ == "__main__":
    main()