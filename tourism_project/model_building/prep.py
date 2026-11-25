import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"

# 1. Load the registered dataset
df = pd.read_csv(RAW_PATH)

# 2. Drop unnecessary identifier column (and unnamed index column if present)
cols_to_drop = [col for col in ["CustomerID", "Unnamed: 0"] if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

# 3. Check and remove duplicate rows
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    print(f"Found {duplicate_count} duplicate rows. Removing them...")
    df.drop_duplicates(inplace=True)
else:
    print("No duplicate rows found.")

# 4. Rectify inconsistent categories in Gender and MaritalStatus
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# 5. Check and handle missing (null) values
null_counts = df.isnull().sum()
total_nulls = null_counts.sum()

if total_nulls > 0:
    print("Missing values found before imputation:")
    print(null_counts[null_counts > 0])

    # Identify numerical and categorical columns (excluding target variable)
    num_cols = df.select_dtypes(include=["number"]).columns.drop("ProdTaken", errors="ignore")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Fill numerical nulls with mean
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    # Fill categorical nulls with mode
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values imputed (mean for numerical, mode for categorical).")
else:
    print("No missing values found.")

# 6. Split features and target (stratified on ProdTaken to preserve class ratio)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Save splits locally as CSVs for the next GitHub Actions job
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation complete: Xtrain.csv, Xtest.csv, ytrain.csv, and ytest.csv saved.")
print("Unique Gender values:", sorted(X["Gender"].unique()))
print("Unique MaritalStatus values:", sorted(X["MaritalStatus"].unique()))
