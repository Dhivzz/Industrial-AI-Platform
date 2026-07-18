def dataset_info(df):
    """
    Display basic information about the dataset.
    """

    print("=" * 50)
    print("DATASET SHAPE")
    print(df.shape)

    print("\n" + "=" * 50)
    print("COLUMN NAMES")
    print(df.columns)

    print("\n" + "=" * 50)
    print("DATA TYPES")
    print(df.dtypes)


def check_missing_values(df):
    """
    Check missing values in each column.
    """

    return df.isnull().sum()


def check_duplicate_rows(df):
    """
    Check duplicate rows.
    """

    return df.duplicated().sum()