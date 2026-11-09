"""
Part A - Instamart Data Foundation
Run from the project root:
    python src/part_a.py
"""

from pathlib import Path
import sqlite3
import pandas as pd

# Project folders
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

CLEAN.mkdir(parents=True, exist_ok=True)


def audit_table(name, df):
    """Print shape, data types, and missing-value counts."""
    print(f"\n--- {name} ---")
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2)
    })
    print(missing)


def main():
    # 1. Load raw files
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    tickets = pd.read_csv(
        RAW / "tickets.csv",
        keep_default_na=False
    )

    print("PART A - RAW DATA AUDIT")

    audit_table("Customers", customers)
    audit_table("Products", products)
    audit_table("Orders", orders)
    audit_table("Tickets", tickets)

    # 2. Required raw audit figures
    print("\n--- REQUIRED RAW AUDIT FIGURES ---")

    amount_numeric = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    delivery_numeric = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    age_numeric = pd.to_numeric(
        customers["age"], errors="coerce"
    )

    print("Raw order rows:", len(orders))
    print("Exact duplicate order rows:", orders.duplicated().sum())
    print("Pending order rows:", (orders["status"] == "pending").sum())
    print("Distinct order IDs:", orders["order_id"].nunique())
    print("Non-numeric order amounts:", amount_numeric.isna().sum())
    print("Negative delivery times:", (delivery_numeric < 0).sum())
    print(
        "Blank delivery times:",
        orders["delivery_minutes"].astype(str).str.strip().eq("").sum()
    )
    print(
        "Blank customer ages:",
        customers["age"].astype(str).str.strip().eq("").sum()
    )
    print(
        "Ages outside 10-100:",
        ((age_numeric < 10) | (age_numeric > 100)).sum()
    )
    print("Distinct raw city spellings:", customers["city"].nunique())
    print(
        "Distinct non-blank gender spellings:",
        customers.loc[
            customers["gender"].astype(str).str.strip() != "",
            "gender"
        ].nunique()
    )
    print(
        "Blank genders:",
        customers["gender"].astype(str).str.strip().eq("").sum()
    )

    # 3. Clean customers
    print("\n--- CLEANING CUSTOMERS ---")

    customers["city"] = (
        customers["city"].astype(str).str.strip().str.title()
    )

    customers["gender"] = (
        customers["gender"].astype(str).str.strip().str.lower()
    )

    gender_map = {
        "male": "male",
        "m": "male",
        "female": "female",
        "f": "female"
    }

    customers["gender"] = (
        customers["gender"].map(gender_map).fillna("unknown")
    )

    customers["age"] = pd.to_numeric(
        customers["age"], errors="coerce"
    )

    customers.loc[
        ~customers["age"].between(10, 100), "age"
    ] = pd.NA

    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians)

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    print("Clean city counts:")
    print(customers["city"].value_counts().sort_index())

    print("\nClean gender counts:")
    print(customers["gender"].value_counts())

    print("\nMissing ages after cleaning:", customers["age"].isna().sum())

    # 4. Clean products
    print("\n--- CLEANING PRODUCTS ---")

    products["price_inr"] = pd.to_numeric(
        products["price_inr"], errors="coerce"
    )

    # 5. Clean orders
    print("\n--- CLEANING ORDERS ---")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    # Remove exact duplicate records
    orders = orders.drop_duplicates().copy()

    # Keep the latest record for each order ID
    orders = orders.sort_values("order_ts")
    orders = orders.drop_duplicates(
        subset="order_id", keep="last"
    ).copy()

    # Convert amount and delivery time to numeric
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # Merge product price and category
    orders = orders.merge(
        products[["product_id", "category", "price_inr"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Fill invalid or missing order amounts from product price
    calculated_amount = (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)

    orders["order_amount"] = orders["order_amount"].fillna(
        calculated_amount
    )

    # Negative delivery times are invalid
    orders.loc[
        orders["delivery_minutes"] < 0,
        "delivery_minutes"
    ] = pd.NA

    # Fill missing delivery times only for completed orders
    completed_mask = orders["status"].eq("completed")

    completed_medians = (
        orders.loc[completed_mask]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )

    orders.loc[completed_mask, "delivery_minutes"] = (
        orders.loc[completed_mask, "delivery_minutes"]
        .fillna(completed_medians)
    )

    # Cancelled orders must retain missing delivery times
    orders.loc[
        orders["status"].eq("cancelled"),
        "delivery_minutes"
    ] = pd.NA

    # Remove helper columns
    orders = orders.drop(columns=["category", "price_inr"])

    # 6. Save clean CSV files
    customers.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)

    orders["order_ts"] = orders["order_ts"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    orders.to_csv(CLEAN / "orders.csv", index=False)

    # 7. Create SQLite database
    print("\n--- CREATING SQLITE DATABASE ---")

    with sqlite3.connect(DB_PATH) as connection:
        customers.to_sql(
            "customers", connection, if_exists="replace", index=False
        )
        products.to_sql(
            "products", connection, if_exists="replace", index=False
        )
        orders.to_sql(
            "orders", connection, if_exists="replace", index=False
        )

    print("Database created:", DB_PATH)

    # 8. Validate reference checkpoints
    completed = orders[orders["status"] == "completed"]
    cancelled = orders[orders["status"] == "cancelled"]

    assert len(customers) == 1200
    assert len(products) == 40
    assert customers["city"].nunique() == 6
    assert customers["gender"].isin(
        ["male", "female", "unknown"]
    ).all()
    assert customers["age"].isna().sum() == 0
    assert orders["order_id"].nunique() == len(orders)
    assert len(orders) == 12309
    assert len(completed) == 11110
    assert len(cancelled) == 1199
    assert not orders["status"].eq("pending").any()
    assert orders["order_amount"].isna().sum() == 0
    assert completed["delivery_minutes"].isna().sum() == 0
    assert cancelled["delivery_minutes"].isna().sum() == 1199

    revenue = round(completed["order_amount"].sum(), 2)
    assert revenue == 2998668.80

    print("\nClean orders:", len(orders))
    print("Completed orders:", len(completed))
    print("Cancelled orders:", len(cancelled))
    print("Completed revenue: INR", f"{revenue:,.2f}")

    print("\nALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()