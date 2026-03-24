import polars as pl
import numpy as np
from datetime import date
from src.config import DATA_PROCESSED, TIMEZONE

VACANCES_VS: list[tuple[date, date]] = [
    (date(2022, 10, 13), date(2022, 10, 23)),
    (date(2022, 12, 23), date(2023, 1, 8)),
    (date(2023, 2, 18), date(2023, 2, 26)),
    (date(2023, 4, 7), date(2023, 4, 16)),
    (date(2023, 6, 24), date(2023, 8, 16)),
    (date(2023, 10, 19), date(2023, 10, 29)),
    (date(2023, 12, 22), date(2024, 1, 7)),
    (date(2024, 2, 10), date(2024, 2, 18)),
    (date(2024, 3, 29), date(2024, 4, 7)),
    (date(2024, 6, 22), date(2024, 8, 14)),
    (date(2024, 10, 14), date(2024, 10, 25)),
    (date(2024, 12, 20), date(2025, 1, 5)),
    (date(2025, 3, 1), date(2025, 3, 9)),
    (date(2025, 4, 18), date(2025, 4, 27)),
    (date(2025, 6, 28), date(2025, 8, 17)),
]


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    local = pl.col("timestamp").dt.convert_time_zone(TIMEZONE)

    df = df.with_columns([
        local.dt.hour().alias("hour"),
        local.dt.minute().alias("minute"),
        local.dt.weekday().alias("weekday"),
        local.dt.month().alias("month"),
        local.dt.year().alias("year"),
        local.dt.ordinal_day().alias("doy"),
    ])

    df = df.with_columns(
        (pl.col("hour") + pl.col("minute") / 60.0).alias("hour_frac")
    )

    for col, period, name in [
        ("hour_frac", 24.0, "hour"),
        ("weekday", 7.0, "weekday"),
        ("month", 12.0, "month"),
        ("doy", 365.25, "doy"),
    ]:
        angle = 2 * np.pi * pl.col(col) / period
        df = df.with_columns([
            angle.sin().alias(f"{name}_sin"),
            angle.cos().alias(f"{name}_cos"),
        ])

    df = df.with_columns(
        (pl.col("weekday") >= 6).cast(pl.Int8).alias("is_weekend")
    )

    try:
        import holidays
        ch = holidays.Switzerland(prov="VS", years=df["year"].unique().to_list())
        holiday_set = set(ch.keys())
        df = df.with_columns(
            local.dt.date()
            .map_elements(lambda d: 1 if d in holiday_set else 0, return_dtype=pl.Int64)
            .alias("is_holiday")
        )
    except ImportError:
        df = df.with_columns(pl.lit(0).alias("is_holiday"))

    df = df.with_columns(
        local.dt.date()
        .map_elements(
            lambda d: 1 if any(s <= d <= e for s, e in VACANCES_VS) else 0,
            return_dtype=pl.Int64)
        .alias("is_school_holiday")
    )

    return df.drop(["minute", "hour_frac", "doy"])


def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    for shift, name in [(1, "1"), (4, "4"), (96, "96"), (672, "672")]:
        df = df.with_columns(pl.col("load").shift(shift).alias(f"load_lag_{name}"))

    for shift, name in [(96, "96"), (192, "192"), (288, "288"), (672, "672")]:
        df = df.with_columns(pl.col("pv_total").shift(shift).alias(f"pv_total_lag_{name}"))

    return df


def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    for col, name in [("load", "load"), ("meteo_temperature_2m", "temp")]:
        if col not in df.columns:
            continue
        for window, label in [(4, "1h"), (12, "3h"), (96, "24h")]:
            df = df.with_columns([
                pl.col(col).rolling_mean(window).alias(f"{name}_rmean_{label}"),
                pl.col(col).rolling_std(window).alias(f"{name}_rstd_{label}"),
            ])
    return df


def add_temperature_gradient(df: pl.DataFrame) -> pl.DataFrame:
    if "meteo_temperature_2m" not in df.columns:
        return df
    temp = pl.col("meteo_temperature_2m")
    df = df.with_columns([
        (temp - temp.shift(4)).alias("temp_gradient_1h"),
        (temp - temp.shift(12)).alias("temp_gradient_3h"),
    ])
    return df


def add_solar_position(df: pl.DataFrame) -> pl.DataFrame:
    try:
        from pvlib import solarposition
        ts = df["timestamp"].to_pandas()
        solar = solarposition.get_solarposition(ts, 46.2333, 7.3592, 500, method="nrel_numpy")
        elev = solar["elevation"].clip(lower=0).values
        df = df.with_columns(pl.Series("solar_elevation", elev, dtype=pl.Float64))
    except ImportError:
        df = df.with_columns(pl.lit(0.0).alias("solar_elevation"))
    return df


def add_pv_trend(df: pl.DataFrame) -> pl.DataFrame:
    ts_min = df["timestamp"].min()
    total_sec = (df["timestamp"].max() - ts_min).total_seconds()
    df = df.with_columns(
        ((pl.col("timestamp") - ts_min).dt.total_seconds() / total_sec)
        .alias("pv_capacity_trend")
    )
    return df


def add_interactions(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col("pv_capacity_trend") > 0.01)
        .then(pl.col("pv_total") / pl.col("pv_capacity_trend"))
        .otherwise(pl.col("pv_total"))
        .alias("pv_normalized")
    )

    if "meteo_temperature_2m" in df.columns:
        temp = pl.col("meteo_temperature_2m")
        df = df.with_columns([
            (temp * pl.col("hour_sin")).alias("temp_x_hour_sin"),
            (temp * pl.col("hour_cos")).alias("temp_x_hour_cos"),
        ])

    if "pred_glob_ctrl" in df.columns:
        df = df.with_columns(
            (pl.col("pred_glob_ctrl") * pl.col("solar_elevation")).alias("pred_radiation_x_elevation")
        )

    if "pred_glob_stde" in df.columns:
        df = df.with_columns(
            (pl.col("pred_glob_stde") * pl.col("pv_total")).alias("pred_glob_uncertainty_x_pv")
        )

    for var in ["pred_t_2m", "pred_glob"]:
        q90, q10 = f"{var}_q90", f"{var}_q10"
        if q90 in df.columns and q10 in df.columns:
            df = df.with_columns(
                (pl.col(q90) - pl.col(q10)).alias(f"{var}_uncertainty_range")
            )

    return df


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_temperature_gradient(df)
    df = add_solar_position(df)
    df = add_pv_trend(df)
    df = add_interactions(df)
    df = df.drop_nulls()
    return df


if __name__ == "__main__":
    df = pl.read_parquet(DATA_PROCESSED / "dataset_merged.parquet")
    df_features = build_features(df)
    df_features.write_parquet(DATA_PROCESSED / "dataset_features.parquet")
    print("Features terminées → dataset_features.parquet")
