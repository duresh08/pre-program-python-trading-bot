import pandas as pd
pd.set_option('display.max_columns', None)

path = "sqlite:///C:\\Users\\dhruv.suresh\\Downloads\\data.db\\data.db"
df = pd.read_sql("SELECT * FROM signals WHERE date BETWEEN '2000-01-01' AND '2001-01-01'", path)
df.head(10)

print("columns are: {}".format(df.columns))
print("\n\n")
print("dtypes: {}".format(df.info()))
print("\n\n")

print("symbol_list: {}".format(list(df['symbol'].unique())))
print("number of unique symbols: {}".format(len(df['symbol'].unique())))

df.describe()

cols_to_drop = ['id','reported_currency']