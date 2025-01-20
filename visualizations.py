import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
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

df = df.drop(columns = cols_to_drop)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df['date'] = pd.to_datetime(df['date'], format = 'mixed')
df['target_date'] = (df['date'] + DateOffset(months=3))

future_df = df.loc[:, ['date', 'symbol', 'adj_close']]
future_df = future_df.rename(columns={'adj_close': 'adj_close_3mo',
                                     'date': 'target_date'})

df = pd.merge(df, future_df, how = 'left', on = ['target_date','symbol'])
df