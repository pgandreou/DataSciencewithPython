import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pycaret.datasets import get_data

pd.set_option('display.max_columns', 30)

#url = 'data/pva_prepared.csv'
#df = pd.read_csv(url)



df = get_data('france')
print(df.head())
