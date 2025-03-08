import polars as pl
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

from matplotlib.pyplot import title
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


df = pl.read_csv("data/data.csv",encoding="iso-8859-1", schema_overrides={"InvoiceNo": pl.Utf8})
print(df.head())

df_clear = df.drop_nulls()
df_clear =df_clear.to_pandas()

#profile = ProfileReport(df_clear, title="Data Profiling")
#profile.to_file("report.html")

df_clear =pl.from_pandas(df_clear)

df_clear = df_clear.group_by("InvoiceDate").agg(
    (pl.col("UnitPrice") * pl.col("Quantity")).sum().alias("sales for day")
)

print(np.std(df_clear))

print(df["InvoiceDate"])

ventas_por_dia = df_clear.with_columns(
    pl.col("InvoiceDate")
    .str.replace(r"(^|\D)(\d{1})/(\d{1})/(\d{4})", r"\10\2/0\3/\4")  # Añadir ceros si es necesario
    .str.strptime(pl.Datetime, "%d/%m/%Y %H:%M",strict=False)  # Convertir a datetime
    .dt.date()  # Extraer solo la fecha (YYYY-MM-DD)
)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(df["InvoiceDate"])
