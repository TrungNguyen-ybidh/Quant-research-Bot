import pandas as pd

df = pd.read_parquet("data/processed/EURUSD_1m_features.parquet")
print(len(df.columns))

