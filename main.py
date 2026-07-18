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
# Load data
df = load_data("data/ai4i2020.csv")

# Preprocessing
dataset_info(df)

print("\nMissing Values")
print(check_missing_values(df))

print("\nDuplicate Rows")
print(check_duplicate_rows(df))

# Feature Engineering
df = create_temperature_difference(df)
df = create_power_feature(df)
df = create_tool_wear_category(df)

print("\nNew Features Created Successfully!")

print(df[[
    "Temperature Difference",
    "Power",
    "Tool Wear Category"
]].head())
#eda 
print("\n")
machine_failure_summary(df)

print("\n")
machine_type_summary(df)