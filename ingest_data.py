import glob
import os

import csv
import sqlite3
import kagglehub

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


def import_csv_to_db(csv_file, table_name):
    # Open the CSV file
    with open(csv_file, 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)

        # Read the header (first row) to create the table
        headers = next(reader)

        # Create a table with the appropriate schema
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(headers)});')

        # Insert data into the table
        for row in reader:
            cursor.execute(f'INSERT INTO {table_name} ({", ".join(headers)}) VALUES ({", ".join(["?"] * len(row))});',
                           row)


# import data to db
csv_files = glob.glob("data/*.csv")
for csv_file, table_name in csv_files:
    import_csv_to_db(csv_file, table_name)

conn.commit()
conn.close()

print(f"Database '{db_name}' created.")
