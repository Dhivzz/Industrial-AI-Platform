import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def prepare_training_data(df):
    """
    Prepare features and target variable.
    """

    # Remove unnecessary columns
    X = df.drop(
        columns=[
            
               "Machine failure",
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
        ]
    )

    y = df["Machine failure"]

    # Convert all categorical/text columns into numbers
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

    return X, y



def train_model(df):
    """
    Train predictive maintenance model.
    """

    X, y = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    print("=" * 50)
    print("MODEL PERFORMANCE")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    return model, X_test, y_test



def save_model(model):
    """
    Save trained model.
    """

    joblib.dump(
        model,
        "models/random_forest_model.pkl"
    )

    print(
        "Model saved successfully!"
    )