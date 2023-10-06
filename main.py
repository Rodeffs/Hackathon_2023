import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data,
                  columns = data.feature_names)
pd.read_parquet('train.parquet', engine='pyarrow')


