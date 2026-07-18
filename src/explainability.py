import matplotlib.pyplot as plt
import pandas as pd
import os


def create_output_folder():
    """
    Create folder for saving explainability results.
    """

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )



def get_feature_importance(model, X_test):
    """
    Display feature importance from Random Forest model.
    """

    importance = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": model.feature_importances_
        }
    )

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("=" * 50)
    print("FEATURE IMPORTANCE")

    print(
        importance.head(10)
    )

    return importance



def plot_feature_importance(model, X_test):
    """
    Plot top important features.
    """

    create_output_folder()

    importance = get_feature_importance(
        model,
        X_test
    )

    top_features = importance.head(10)

    plt.figure(
        figsize=(8, 5)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.title(
        "Top 10 Features Influencing Machine Failure"
    )

    plt.gca().invert_yaxis()

    plt.savefig(
        "reports/figures/explainability_feature_importance.png",
        bbox_inches="tight"
    )

    plt.close()



def failure_reason(model, X_sample):
    """
    Explain prediction for one machine sample.
    """

    prediction = model.predict(
        X_sample
    )[0]

    probability = model.predict_proba(
        X_sample
    )[0][1]


    print("=" * 50)

    if prediction == 1:
        print(
            "Prediction: MACHINE FAILURE RISK"
        )
    else:
        print(
            "Prediction: MACHINE HEALTHY"
        )


    print(
        "Failure Probability:",
        round(probability * 100, 2),
        "%"
    )


    feature_values = pd.DataFrame(
        {
            "Feature": X_sample.columns,
            "Value": X_sample.iloc[0].values
        }
    )

    print("\nMachine Condition:")
    print(feature_values)