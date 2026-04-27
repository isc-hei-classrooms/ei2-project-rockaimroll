import polars as pl
from pathlib import Path

# Adapte ce path si besoin
df = pl.read_parquet("data/processed/dataset_normalized.parquet")
df = df.filter(pl.col("forecast_load").is_not_null()
               & pl.col("load").is_not_null())

# Convertir en local Zurich pour decoupage saisonnier
df = df.with_columns(
    pl.col("timestamp").dt.convert_time_zone("Europe/Zurich").alias("ts_local")
)
df = df.with_columns(pl.col("ts_local").dt.month().alias("month"))

# Pour chaque saison, calculer MAE pour different shifts (-4..+4 pas de 15min)
# Un shift de -4 pas = -1h
def mae_for_shift(df_sub, shift_pas):
    if shift_pas == 0:
        diff = (df_sub["load"] - df_sub["forecast_load"]).abs().mean()
    else:
        # Shift forecast_load de shift_pas pas vers le futur
        df_shifted = df_sub.with_columns(
            pl.col("forecast_load").shift(shift_pas).alias("forecast_shifted")
        ).drop_nulls("forecast_shifted")
        diff = (df_shifted["load"] - df_shifted["forecast_shifted"]).abs().mean()
    return float(diff)

print("Test shift baseline pour minimiser MAE")
print(f"{'shift_pas':>10} {'shift_min':>10} {'MAE_hiver':>12} {'MAE_ete':>12}")
df_winter = df.filter(pl.col("month").is_in([12, 1, 2]))
df_summer = df.filter(pl.col("month").is_in([6, 7, 8]))

for shift in range(-4, 5):
    mae_w = mae_for_shift(df_winter, shift)
    mae_s = mae_for_shift(df_summer, shift)
    marker_w = " <-- min" if shift == 0 else ""
    print(f"{shift:>10} {shift*15:>9}m {mae_w:>12.4f} {mae_s:>12.4f}")