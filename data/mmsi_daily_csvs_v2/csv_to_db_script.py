import glob
import pandas as pd

files = glob.glob("./*/*.csv", recursive=True)
for file in files:
    df_csv = pd.read_csv(file)
    print(df_csv)

    break
