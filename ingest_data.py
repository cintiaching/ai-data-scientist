import glob
import os

import sqlite3
import kagglehub
import pandas as pd

from dotenv import load_dotenv

load_dotenv(".env")
os.makedirs("data", exist_ok=True)

# Download data
path = kagglehub.dataset_download("dataceo/sales-and-customer-data")
print("Path to dataset files:", path)

# create a sqlite database
db_name = "data/sales-and-customer-database.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()


def infer_datatype(series):
    """Infer the data type of a pandas Series."""
    if pd.api.types.is_integer_dtype(series):
        return 'INTEGER'
    elif pd.api.types.is_float_dtype(series):
        return 'REAL'
    elif pd.api.types.is_bool_dtype(series):
        return 'BOOLEAN'
    else:
        return 'TEXT'


def import_csv_to_db(csv_file, table_name):
    df = pd.read_csv(csv_file)
    column_types = {col: infer_datatype(df[col]) for col in df.columns}
    # Create table with inferred types
    columns_with_types = ', '.join(f"{col} {dtype}" for col, dtype in column_types.items())
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_with_types});"
    cursor.execute(create_table_query)
    # Insert data into the table
    df.to_sql(table_name, conn, if_exists='append', index=False)


# import data to db
csv_files = glob.glob("data/**/*.csv", recursive=True)
for csv_file in csv_files:
    table_name = os.path.basename(csv_file).split(".")[0]
    import_csv_to_db(csv_file, table_name)

conn.commit()
conn.close()

print(f"Database '{db_name}' created.")
