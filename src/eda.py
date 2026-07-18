import pandas as pd


def machine_failure_summary(df):
    """
    Display machine failure counts and percentages.
    """

    print("=" * 50)
    print("MACHINE FAILURE SUMMARY")

    counts = df["Machine failure"].value_counts()
    percentages = df["Machine failure"].value_counts(normalize=True) * 100

    print("\nCounts:")
    print(counts)

    print("\nPercentages:")
    print(percentages.round(2))


def machine_type_summary(df):
    """
    Analyze failures by machine type.
    """

    print("\n" + "=" * 50)
    print("FAILURE RATE BY MACHINE TYPE")

    result = df.groupby("Type")["Machine failure"].mean() * 100

    print(result.round(2))