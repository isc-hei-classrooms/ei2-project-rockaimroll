import polars as pl
from src.config import DATA_RAW, DATA_PROCESSED, TIMEZONE


def to_utc_oiken(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone(TIMEZONE, ambiguous="latest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .alias("timestamp")
    )
    df = df.filter(pl.col("timestamp").is_not_null())
    df = df.unique(subset=["timestamp"], keep="first").sort("timestamp")
    return df


def to_utc_meteo(df: pl.DataFrame) -> pl.DataFrame:
    if df["timestamp"].dtype == pl.Datetime("ns", "UTC"):
        return df
    try:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    except Exception:
        df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    return df


def resample_15min(df: pl.DataFrame, value_cols: list[str]) -> pl.DataFrame:
    grid = pl.DataFrame({
        "timestamp": pl.datetime_range(
            df["timestamp"].min(), df["timestamp"].max(),
            interval="15m", eager=True
        )
    })
    df = grid.join(df, on="timestamp", how="left")
    for col in value_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).interpolate().alias(col))
    return df


def normalize_oiken(df: pl.DataFrame) -> pl.DataFrame:
    df = to_utc_oiken(df)
    if df["forecast_load"].null_count() > 0:
        df = df.with_columns(pl.col("forecast_load").interpolate().alias("forecast_load"))
    return df


def normalize_meteo(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    df = to_utc_meteo(df)
    value_cols = [c for c in df.columns if c not in ("timestamp", "site")]

    site_frames = []
    for site in df["site"].unique().to_list():
        df_site = df.filter(pl.col("site") == site).drop("site")
        df_site = resample_15min(df_site, value_cols)
        site_frames.append(df_site)

    df_all = pl.concat(site_frames)
    df_agg = (
        df_all.group_by("timestamp")
        .agg([pl.col(c).mean() for c in value_cols])
        .sort("timestamp")
    )

    if prefix:
        df_agg = df_agg.rename({c: f"{prefix}{c}" for c in value_cols})

    return df_agg


def merge_all(oiken, meteo_real, meteo_pred):
    merged = oiken.join(meteo_real, on="timestamp", how="left")
    merged = merged.join(meteo_pred, on="timestamp", how="left")
    return merged


if __name__ == "__main__":
    oiken = pl.read_parquet(DATA_RAW / "oiken_clean.parquet")
    meteo_real = pl.read_parquet(DATA_RAW / "meteo_real.parquet")
    meteo_pred = pl.read_parquet(DATA_RAW / "meteo_pred.parquet")

    oiken_norm = normalize_oiken(oiken)
    meteo_real_norm = normalize_meteo(meteo_real, prefix="meteo_")
    meteo_pred_norm = normalize_meteo(meteo_pred, prefix="")

    dataset = merge_all(oiken_norm, meteo_real_norm, meteo_pred_norm)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(DATA_PROCESSED / "dataset_merged.parquet")
    print("Normalisation terminée → dataset_merged.parquet")
