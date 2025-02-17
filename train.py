import os
import datetime

from agents.data_analyst import DataAnalystVanna


def train(vn):
    df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
    for ddl in df_ddl["sql"].to_list():
        vn.train(ddl=ddl)

    # Sometimes you may want to add documentation about your business terminology or definitions.
    vn.train(
        documentation="Our business defines financial year start with april to mar of each year")
    vn.train(
        documentation="The invoice_date of sales_data is in dd-MM-yyyy format")
    vn.train(
        documentation="Today's date is {}".format(datetime.date.today()))
    # At any time you can inspect what training data the package is able to reference
    training_data = vn.get_training_data()
    print("training_data", training_data)

    print("Training is completed.")


if __name__ == "__main__":
    vn = DataAnalystVanna(config={"model": "gpt-4o-mini", "client": "persistent", "path": "./vanna-db"})
    vn.connect_to_sqlite(f"{os.getenv("SQLITE_DATABASE_NAME", "data/sales-and-customer-database.db")}")
    train(vn=vn)
