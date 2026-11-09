from pathlib import Path
import sqlite3
import json

ROOT = Path(__file__).resolve().parents[1]

required_files = [
    "generate_data.py",
    "mock_llm.py",
    "requirements.txt",
    "README.md",
    "data/instamart.db",
    "src/part_a.py",
    "src/part_b.py",
    "src/part_c.py",
    "src/part_d.py",
    "src/run_sql.py",
    "sql/queries.sql",
    "outputs/customer_segments.csv",
    "outputs/retention_list.csv",
    "outputs/lapse_model.joblib",
    "outputs/triage_results.csv",
    "outputs/triage_budget_demo.csv",
    "outputs/agent_runs.jsonl",
    "outputs/refund_ledger.jsonl",
]

print("INSTAMART CAPSTONE - FINAL VALIDATION")
print("=" * 45)

missing = []

for filename in required_files:
    if (ROOT / filename).exists():
        print(f"PASS: {filename}")
    else:
        print(f"MISSING: {filename}")
        missing.append(filename)

print("\nCHECKING DATABASE")

db_path = ROOT / "data" / "instamart.db"

if db_path.exists():
    connection = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    connection.close()

    for table in ["customers", "products", "orders"]:
        if table in tables:
            print(f"PASS: {table} table")
        else:
            print(f"MISSING: {table} table")
            missing.append(f"database table: {table}")

print("\nFINAL RESULT")
print("=" * 45)

if missing:
    print("VALIDATION NEEDS ATTENTION")
    print("Missing items:", missing)
else:
    print("ALL REQUIRED FILE CHECKS PASSED!")