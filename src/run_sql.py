import sqlite3
import pandas as pd
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

with open(SQL_PATH, "r", encoding="utf-8") as file:
    sql_text = file.read()

queries = re.split(r"(?=-- Q[1-7]:)", sql_text)

connection = sqlite3.connect(DB_PATH)

for query in queries:
    query = query.strip()

    if not query:
        continue

    title = query.splitlines()[0]

    sql = "\n".join(
        line for line in query.splitlines()
        if not line.strip().startswith("--")
    ).strip()

    if sql:
        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)

        result = pd.read_sql_query(sql, connection)
        print(result.to_string(index=False))
        print("\nRows returned:", len(result))

connection.close()