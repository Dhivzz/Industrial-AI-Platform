import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model performance.
    """

    predictions = model.predict(X_test)

    print("=" * 50)
    print("CONFUSION MATRIX")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(cm)


    print("\nROC-AUC SCORE")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        round(auc, 3)
    )


def plot_confusion_matrix(model, X_test, y_test):
    """
    Save confusion matrix plot.
    """

    cm = confusion_matrix(
        y_test,
        model.predict(X_test)
    )

    plt.figure(
        figsize=(6,4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.savefig(
        "reports/figures/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()



def plot_feature_importance(model, X_test):
    """
    Plot important features.
    """

    importance = model.feature_importances_

    features = X_test.columns

    feature_importance = (
        importance
        .argsort()[-10:]
    )

    plt.figure(
        figsize=(8,5)
    )

    plt.barh(
        features[feature_importance],
        importance[feature_importance]
    )

    plt.title(
        "Top 10 Important Features"
    )

    plt.savefig(
        "reports/figures/feature_importance.png",
        bbox_inches="tight"
    )

    plt.close()