from src.data_loader import load_data

from src.preprocessing import (
    dataset_info,
    check_missing_values,
    check_duplicate_rows
)

from src.feature_engineering import (
    create_temperature_difference,
    create_power_feature,
    create_tool_wear_category
)

from src.eda import (
    machine_failure_summary,
    machine_type_summary
)

from src.visualization import (
    plot_failure_distribution,
    plot_failure_by_type,
    plot_correlation_heatmap,
    plot_feature_boxplots
)

from src.train_model import (
    train_model,
    save_model
)

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance
)

from src.explainability import (
    plot_feature_importance as plot_xai_feature_importance,
    failure_reason
)


# =========================
# LOAD DATA
# =========================

df = load_data(
    "data/ai4i2020.csv"
)


# =========================
# PREPROCESSING
# =========================

dataset_info(df)

print("\nMissing Values")
print(
    check_missing_values(df)
)

print("\nDuplicate Rows")
print(
    check_duplicate_rows(df)
)


# =========================
# FEATURE ENGINEERING
# =========================

df = create_temperature_difference(df)

df = create_power_feature(df)

df = create_tool_wear_category(df)


print("\nNew Features Created Successfully!")

print(
    df[
        [
            "Temperature Difference",
            "Power",
            "Tool Wear Category"
        ]
    ].head()
)


# =========================
# EDA
# =========================

print("\n")
machine_failure_summary(df)

print("\n")
machine_type_summary(df)


# =========================
# VISUALIZATION
# =========================

plot_failure_distribution(df)

plot_failure_by_type(df)

plot_correlation_heatmap(df)

plot_feature_boxplots(df)


# =========================
# MODEL TRAINING
# =========================

model, X_test, y_test = train_model(df)

save_model(model)


# =========================
# MODEL EVALUATION
# =========================

evaluate_model(
    model,
    X_test,
    y_test
)

plot_confusion_matrix(
    model,
    X_test,
    y_test
)

plot_feature_importance(
    model,
    X_test
)


# =========================
# EXPLAINABLE AI
# =========================

plot_xai_feature_importance(
    model,
    X_test
)

failure_reason(
    model,
    X_test.iloc[[0]]
)