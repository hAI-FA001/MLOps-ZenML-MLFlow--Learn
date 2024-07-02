import pandas as pd
import os
from functools import reduce

csv_files = []
for csv_name in os.listdir("./data"):
    csv_file = pd.read_csv(f"./data/{csv_name}")
    csv_files.append(csv_file)

single_csv_file = reduce(lambda single_csv, current_csv: pd.concat([single_csv, current_csv], axis=1), csv_files)

single_csv_file.to_csv("./data/olist_merged.csv")