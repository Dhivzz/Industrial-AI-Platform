import pandas as pd


def create_temperature_difference(df):
    """
    Create a new feature:
    Process Temperature - Air Temperature
    """

    df["Temperature Difference"] = (
        df["Process temperature [K]"] -
        df["Air temperature [K]"]
    )

    return df


def create_power_feature(df):
    """
    Approximate machine power using
    Torque × Rotational Speed
    """

    df["Power"] = (
        df["Torque [Nm]"] *
        df["Rotational speed [rpm]"]
    )

    return df


def create_tool_wear_category(df):
    """
    Categorize tool wear into Low, Medium, High.
    """

    df["Tool Wear Category"] = pd.cut(
        df["Tool wear [min]"],
        bins=[0, 100, 200, 300],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    return df