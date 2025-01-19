import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

data_location = 'sqlite:///D:\\pre-program-python-trading-bot\\data.db\\data.db'
df = pd.read_sql("SELECT * FROM signals WHERE date BETWEEN '2000-01-01' AND '2005-12-31'", data_location)
df