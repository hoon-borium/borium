import pandas as pd
df = pd.read_excel("3KOBROS Limited_RAW-REPORT_2025-01-01_2025-11-04.xlsx")
df.to_parquet("3KOBROS.parquet")
print("convert completedd!")
