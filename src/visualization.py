import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_output_folder():
    """
    Create folder for saving figures.
    """

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )


def plot_failure_distribution(df):
    """
    Plot machine failure distribution.
    """

    create_output_folder()

    plt.figure(figsize=(6, 4))

    sns.countplot(
        data=df,
        x="Machine failure"
    )

    plt.title("Machine Failure Distribution")

    plt.savefig(
        "reports/figures/failure_distribution.png",
        bbox_inches="tight"
    )

    plt.close()


def plot_failure_by_type(df):
    """
    Plot failure rate by machine type.
    """

    create_output_folder()

    failure_rate = (
        df.groupby("Type")["Machine failure"]
        .mean()
        * 100
    )

    plt.figure(figsize=(6, 4))

    failure_rate.plot(
        kind="bar"
    )

    plt.ylabel("Failure Rate (%)")
    plt.title("Failure Rate by Machine Type")

    plt.savefig(
        "reports/figures/failure_by_type.png",
        bbox_inches="tight"
    )

    plt.close()


def plot_correlation_heatmap(df):
    """
    Generate correlation heatmap.
    """

    create_output_folder()

    plt.figure(figsize=(10, 8))

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    correlation = numeric_df.corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")

    plt.savefig(
        "reports/figures/correlation_heatmap.png",
        bbox_inches="tight"
    )

    plt.close()


def plot_feature_boxplots(df):
    """
    Generate boxplots for numerical features.
    """

    create_output_folder()

    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    plt.figure(figsize=(12, 6))

    df[features].boxplot()

    plt.xticks(
        rotation=45
    )

    plt.title("Feature Distribution Boxplots")

    plt.savefig(
        "reports/figures/feature_boxplots.png",
        bbox_inches="tight"
    )

    plt.close()