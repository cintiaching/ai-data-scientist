from agents.data_analyst import DataAnalyst

vn = DataAnalyst(config={"model": "gpt-4o-mini"})
vn.connect_to_sqlite("data/sales-and-customer-database.db")

# Training
# You only need to train once. Do not train again unless you want to add more training data.
df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
for ddl in df_ddl["sql"].to_list():
    vn.train(ddl=ddl)


# Sometimes you may want to add documentation about your business terminology or definitions.
vn.train(
    documentation="Our business defines financial year start with april to mar of each year")


# At any time you can inspect what training data the package is able to reference
training_data = vn.get_training_data()
print("training_data", training_data)

print("Training is completed.")
