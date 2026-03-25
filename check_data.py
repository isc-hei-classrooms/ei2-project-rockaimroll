import polars as pl
from src.config import DATA_RAW, DATA_PROCESSED

oiken = pl.read_parquet(DATA_RAW / "oiken_clean.parquet")
meteo_r = pl.read_parquet(DATA_RAW / "meteo_real.parquet")
meteo_p = pl.read_parquet(DATA_RAW / "meteo_pred.parquet")
print("=== ACQUISITION ===")
print(f"OIKEN     : {oiken.shape[0]} lignes, {oiken.shape[1]} col, nulls load={oiken['load'].null_count()}")
print(f"Meteo real: {meteo_r.shape[0]} lignes, {meteo_r.shape[1]} col")
print(f"Meteo pred: {meteo_p.shape[0]} lignes, {meteo_p.shape[1]} col")

merged = pl.read_parquet(DATA_PROCESSED / "dataset_merged.parquet")
print("\n=== NORMALISATION ===")
print(f"Merged    : {merged.shape[0]} lignes, {merged.shape[1]} col")
print(f"Nulls load: {merged['load'].null_count()}")
print(f"Plage     : {merged['timestamp'].min()} -> {merged['timestamp'].max()}")

feat = pl.read_parquet(DATA_PROCESSED / "dataset_features.parquet")
print("\n=== FEATURES ===")
print(f"Features  : {feat.shape[0]} lignes, {feat.shape[1]} col")
print(f"Nulls     : {feat.null_count().sum_horizontal().item()}")
print(f"Lignes perdues (lags) : {merged.shape[0] - feat.shape[0]}")