import sqlite3
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "data" / "clean"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Part B - Customer Intelligence")
print("Database:", DB_PATH)
# -----------------------------------------
# 1. RFM CUSTOMER ANALYSIS
# -----------------------------------------

print("\n--- RFM CUSTOMER ANALYSIS ---")


orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders = orders[orders["status"] == "completed"].copy()
reference_date = pd.Timestamp("2025-10-01 00:00:00")
rfm = orders.groupby("customer_id").agg(
    recency=("order_ts", lambda x: (reference_date - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

print(rfm.head())

rfm.to_csv(OUTPUT_DIR / "rfm_analysis.csv", index=False)

print("\nRFM analysis saved successfully.")
print("Total customers:", len(rfm))

# 2. CUSTOMER SEGMENTATION USING K-MEANS

print("\n--- CUSTOMER SEGMENTATION ---")

features = rfm[["recency", "frequency", "monetary"]].copy()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

print("\n--- PCA ANALYSIS ---")
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", pca.explained_variance_ratio_.sum())

print("\n--- K-MEANS EVALUATION ---")

from sklearn.metrics import silhouette_score

for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_features)

    inertia = model.inertia_
    silhouette = silhouette_score(scaled_features, labels)

    print(
        f"K={k}: Inertia={inertia:.2f}, "
        f"Silhouette={silhouette:.3f}"
    )

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["segment"] = kmeans.fit_predict(scaled_features)
segment_names = {
    0: "Regular Low-Value Customers",
    1: "Premium Customers",
    2: "Loyal Frequent Customers",
    3: "At-Risk Low-Engagement Customers"
}

rfm["segment_name"] = rfm["segment"].map(segment_names)

premium_segment = rfm.groupby("segment")["monetary"].mean().idxmax()
rfm["is_premium"] = rfm["segment"] == premium_segment

# Create the required output folder
FINAL_OUTPUT_DIR = ROOT / "outputs"
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Assign meaningful customer segment names
segment_names = {
    0: "Occasional Customers",
    1: "Premium Customers",
    2: "Loyal Customers",
    3: "At Risk Customers"
}

rfm["segment_id"] = rfm["segment"]
rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Identify the Premium segment using highest average monetary value
premium_segment = rfm.groupby("segment_id")["monetary"].mean().idxmax()
rfm["is_premium"] = rfm["segment_id"] == premium_segment

# Save the required columns
required_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
]

rfm[required_columns].to_csv(
    FINAL_OUTPUT_DIR / "customer_segments.csv",
    index=False
)

print("Final customer segments saved successfully!")
print("\nCustomer segments created successfully.")

# =====================================================
# 3. CUSTOMER LAPSE PREDICTION
# =====================================================

print("\n--- CUSTOMER LAPSE PREDICTION ---")

# Load clean datasets
# Load clean datasets
customers = pd.read_csv(ROOT / "data" / "clean" / "customers.csv")
orders = pd.read_csv(ROOT / "data" / "clean" / "orders.csv")
products = pd.read_csv(ROOT / "data" / "clean" / "products.csv")

print("\nCustomer columns:")
print(customers.columns.tolist())

print("\nOrder columns:")
print(orders.columns.tolist())

print("\nProduct columns:")
print(products.columns.tolist())

print("\nCustomer dataset shape:", customers.shape)
print("Order dataset shape:", orders.shape)
print("Product dataset shape:", products.shape)

# ==========================================
# PREPARE CUSTOMER LAPSE DATASET
# ==========================================

print("\nPreparing customer lapse dataset...")

# Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Define cutoff and label period
cutoff_date = pd.Timestamp("2025-09-01")
label_end_date = pd.Timestamp("2025-10-01")

# Orders before the cutoff
past_orders = orders[
    (orders["order_ts"] < cutoff_date)
]

# Use only completed orders for customer behaviour
completed_past = past_orders[
    past_orders["status"] == "completed"
].copy()

# Customers who placed at least one completed order before cutoff
cohort_ids = completed_past["customer_id"].unique()

# Create lapse labels using September orders
september_completed = orders[
    (orders["order_ts"] >= cutoff_date) &
    (orders["order_ts"] < label_end_date) &
    (orders["status"] == "completed")
]

active_ids = set(september_completed["customer_id"])

lapse_data = pd.DataFrame({
    "customer_id": cohort_ids
})

lapse_data["lapsed"] = (
    ~lapse_data["customer_id"].isin(active_ids)
).astype(int)

print("Cohort customers:", len(lapse_data))
print("Lapsed customers:", lapse_data["lapsed"].sum())
print("Lapse rate:", round(lapse_data["lapsed"].mean() * 100, 2), "%")


# ==========================================
# CREATE CUSTOMER BEHAVIOUR FEATURES
# ==========================================

print("\nCreating customer behaviour features...")

# Completed orders before cutoff
completed = completed_past.copy()

# Calculate RFM and other purchase features
features = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean")
).reset_index()

# Recency in days from cutoff date
features["recency"] = (
    cutoff_date - features["last_order"]
).dt.days

# Cancellation rate using all final-status orders before cutoff
final_orders = past_orders[
    past_orders["status"].isin(["completed", "cancelled"])
].copy()

cancellation = final_orders.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    cancelled_orders=("status", lambda x: (x == "cancelled").sum())
).reset_index()

cancellation["cancellation_rate"] = (
    cancellation["cancelled_orders"] / cancellation["total_orders"]
)

# Find each customer's favourite product category
category_orders = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite_category = (
    category_orders.groupby("customer_id")["category"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reset_index(name="favourite_category")
)

# Combine all features
features = features.merge(
    cancellation[["customer_id", "cancellation_rate"]],
    on="customer_id",
    how="left"
)

features = features.merge(
    favourite_category,
    on="customer_id",
    how="left"
)

features = features.merge(
    customers[["customer_id", "city", "age", "gender", "signup_date"]],
    on="customer_id",
    how="left"
)

# Customer tenure in days
features["tenure_days"] = (
    cutoff_date - features["signup_date"]
).dt.days

# Add lapse labels
features = features.merge(
    lapse_data,
    on="customer_id",
    how="inner"
)

print("\nCustomer feature dataset created!")
print("Feature dataset shape:", features.shape)
print(features.head())
# ==========================================
# TRAIN-TEST SPLIT AND PREPROCESSING
# ==========================================

print("\nPreparing training and testing data...")

# Select model features
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

# Check required columns
model_columns = numeric_features + categorical_features

print("\nMissing feature columns:")
print(set(model_columns) - set(features.columns))

# Create X and y
X = features[model_columns].copy()
y = features["lapsed"].copy()

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Create preprocessing steps
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed!")
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

# Show frequency mean learned by training scaler
frequency_position = numeric_features.index("frequency")

print(
    "Training frequency mean:",
    preprocessor.named_transformers_["num"].mean_[frequency_position]
)

print("Full cohort frequency mean:", X["frequency"].mean())
     
# ==========================================
#  MODEL TRAINING AND HYPERPARAMETER TUNING
# ==========================================

print("\n--- LOGISTIC REGRESSION TUNING ---")

# Build Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Tune C parameter
lr_parameters = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

lr_grid = GridSearchCV(
    lr_pipeline,
    lr_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)

print("Best Logistic Regression parameters:", lr_grid.best_params_)
print("Best Logistic Regression CV F1:", lr_grid.best_score_)


print("\n--- RANDOM FOREST TUNING ---")

# Build Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Tune Random Forest parameters
rf_parameters = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("Best Random Forest parameters:", rf_grid.best_params_)
print("Best Random Forest CV F1:", rf_grid.best_score_)
# ==========================================
# LOGISTIC REGRESSION WITH OVERSAMPLING
# ==========================================

print("\n--- LOGISTIC REGRESSION WITH OVERSAMPLING ---")

# Convert training labels to NumPy arrays
y_train_array = np.asarray(y_train)

# Find the two classes
not_lapsed_indices = np.where(y_train_array == 0)[0]
lapsed_indices = np.where(y_train_array == 1)[0]

# Duplicate lapsed customers until both classes are balanced
extra_indices = np.random.RandomState(42).choice(
    lapsed_indices,
    size=len(not_lapsed_indices) - len(lapsed_indices),
    replace=True
)

# Keep all original training rows and add duplicated rows
X_train_balanced = np.vstack([
    X_train_processed,
    X_train_processed[extra_indices]
])

y_train_balanced = np.concatenate([
    y_train_array,
    y_train_array[extra_indices]
])

print("Balanced training class counts:")
print(pd.Series(y_train_balanced).value_counts())

# Train Logistic Regression on balanced training data
balanced_lr = LogisticRegression(
    C=lr_grid.best_params_["classifier__C"],
    max_iter=1000
)

balanced_lr.fit(X_train_balanced, y_train_balanced)

# Evaluate on the original, unchanged test data
balanced_predictions = balanced_lr.predict(X_test_processed)

print("\nOversampled Logistic Regression Results:")
print(confusion_matrix(y_test, balanced_predictions))
print(classification_report(
    y_test,
    balanced_predictions,
    zero_division=0
))

# ==========================================
# FINAL MODEL SELECTION AND FEATURE IMPORTANCE
# ==========================================

print("\n--- FINAL MODEL SELECTION ---")

# Select Random Forest based on test F1 score
selected_model = rf_grid.best_estimator_
print("Selected model: Random Forest")
rf_predictions = selected_model.predict(X_test)
print("Test F1 score:", f1_score(y_test, rf_predictions))

# Random Forest feature importance
rf_classifier = selected_model.named_steps["classifier"]
feature_names = selected_model.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_classifier.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 5 Random Forest Features:")
print(importance_df.head(5))

# Logistic Regression coefficient interpretation
lr_classifier = lr_grid.best_estimator_.named_steps["classifier"]
lr_feature_names = lr_grid.best_estimator_.named_steps["preprocessor"].get_feature_names_out()

coefficient_df = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_classifier.coef_[0]
})

print("\nThree Logistic Regression Coefficients:")
print(
    coefficient_df.reindex(
        coefficient_df["coefficient"].abs()
        .sort_values(ascending=False).index
    ).head(3)
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_path = FINAL_OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(selected_model, model_path)

print("\nModel saved:", model_path)

# Reload and verify predictions
loaded_model = joblib.load(model_path)

saved_predictions = loaded_model.predict(X_test)

print(
    "Model reload prediction verification:",
    np.array_equal(rf_predictions, saved_predictions)
)

# ==========================================
# CREATE TOP 10 CUSTOMER RETENTION LIST
# ==========================================

print("\n--- PREMIUM CUSTOMER RETENTION LIST ---")

# Identify Premium customers from whole-window RFM segmentation
premium_ids = set(
    rfm.loc[rfm["is_premium"], "customer_id"]
)

# Identify Premium customers in the test cohort
retention_candidates = features[
    features["customer_id"].isin(premium_ids)
    & features["customer_id"].isin(X_test.index)
].copy()

# Use test customer predictions and probabilities
test_results = pd.DataFrame({
    "customer_id": X_test.index,
    "lapse_probability": selected_model.predict_proba(X_test)[:, 1],
    "actually_lapsed": y_test.values
})

# Correctly attach customer IDs from the test feature rows
test_results["customer_id"] = features.loc[
    X_test.index, "customer_id"
].values

# Keep Premium customers only
retention_list = test_results[
    test_results["customer_id"].isin(premium_ids)
].copy()

# Select the ten highest lapse probabilities
retention_list = retention_list.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention_list.to_csv(
    FINAL_OUTPUT_DIR / "retention_list.csv",
    index=False
)

print("\nTop 10 retention list:")
print(retention_list)

print(
    "\nActually lapsed customers in top 10:",
    retention_list["actually_lapsed"].sum()
)

print("\nPART B FINAL OUTPUTS CREATED!")