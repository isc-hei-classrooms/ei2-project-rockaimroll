
import polars as pl
import numpy as np
from datetime import date

from src.config import DATA_PROCESSED, TIMEZONE


# ============================================================
# Vacances scolaires Valais
# ============================================================

VACANCES_SCOLAIRES_VS: list[tuple[date, date]] = [
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


def _is_school_holiday(d: date) -> bool:
    for start, end in VACANCES_SCOLAIRES_VS:
        if start <= d <= end:
            return True
    return False


# ============================================================
# 1. Variables calendaires + feries + ponts
# ============================================================

def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    ts_local = pl.col("timestamp").dt.convert_time_zone(TIMEZONE)

    df = df.with_columns([
        ts_local.dt.hour().alias("hour"),
        ts_local.dt.weekday().alias("weekday"),
        ts_local.dt.month().alias("month"),
        ts_local.dt.year().alias("year"),
        ts_local.dt.ordinal_day().alias("day_of_year"),
    ])

    df = df.with_columns([
        (2 * np.pi * pl.col("hour") / 24).sin().alias("hour_sin"),
        (2 * np.pi * pl.col("hour") / 24).cos().alias("hour_cos"),
        (2 * np.pi * pl.col("weekday") / 7).sin().alias("weekday_sin"),
        (2 * np.pi * pl.col("weekday") / 7).cos().alias("weekday_cos"),
        (2 * np.pi * pl.col("month") / 12).sin().alias("month_sin"),
        (2 * np.pi * pl.col("month") / 12).cos().alias("month_cos"),
        (2 * np.pi * pl.col("day_of_year") / 365).sin().alias("doy_sin"),
        (2 * np.pi * pl.col("day_of_year") / 365).cos().alias("doy_cos"),
    ])

    df = df.with_columns(
        pl.when(pl.col("weekday") >= 6).then(1).otherwise(0)
        .cast(pl.Int64).alias("is_weekend")
    )

    # Jours feries suisses (canton du Valais)
    try:
        import holidays
        years = df["year"].unique().to_list()
        ch_holidays = holidays.Switzerland(prov="VS", years=years)
        holiday_dates = set(ch_holidays.keys())

        df = df.with_columns(
            ts_local.dt.date().alias("_date")
        )
        df = df.with_columns(
            pl.col("_date").map_elements(
                lambda d: 1 if d in holiday_dates else 0,
                return_dtype=pl.Int64
            ).alias("is_holiday")
        )
        df = df.drop("_date")
    except ImportError:
        print("    Warning: 'holidays' non installe, is_holiday = 0")
        df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("is_holiday"))

    # Vacances scolaires valaisannes
    df = df.with_columns(
        ts_local.dt.date().alias("_date_sh")
    )
    df = df.with_columns(
        pl.col("_date_sh").map_elements(
            lambda d: 1 if _is_school_holiday(d) else 0,
            return_dtype=pl.Int64
        ).alias("is_school_holiday")
    )
    df = df.drop("_date_sh")

    # Jours ponts (lundi apres ferie ou vendredi avant ferie)
    df = df.with_columns([
        pl.col("is_holiday").shift(96).fill_null(0).alias("_holiday_yesterday"),
        pl.col("is_holiday").shift(-96).fill_null(0).alias("_holiday_tomorrow"),
    ])
    df = df.with_columns(
        pl.when(
            (pl.col("is_holiday") == 0)
            & (pl.col("is_weekend") == 0)
            & (
                ((pl.col("weekday") == 5) & (pl.col("_holiday_yesterday") == 1))
                | ((pl.col("weekday") == 1) & (pl.col("_holiday_tomorrow") == 1))
            )
        ).then(1).otherwise(0)
        .cast(pl.Int64).alias("is_bridge_day")
    )
    df = df.drop(["_holiday_yesterday", "_holiday_tomorrow"])

    # Jour apres repos (lundi ou lendemain de ferie)
    df = df.with_columns(
        pl.col("is_holiday").shift(96).fill_null(0).alias("_hol_prev")
    )
    df = df.with_columns(
        pl.when(
            (pl.col("_hol_prev") == 1) | (pl.col("weekday") == 1)
        ).then(1).otherwise(0)
        .cast(pl.Int64).alias("is_day_after_rest")
    )
    df = df.drop("_hol_prev")

    n_bridge = df["is_bridge_day"].sum()
    n_school = df["is_school_holiday"].sum()
    print(f"    Vacances scolaires: {n_school} pas | Ponts: {n_bridge} pas")

    return df


# ============================================================
# 2. Lag features (hebdomadaires + J-2/J-3 uniquement)
# ============================================================

def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """Lags sur charge et PV.

    Seuls les lags strictement disponibles a 11h le jour J sont crees.
    load_lag_96 est cree car il sert de base a des features derivees
    (load_recent_mean_2d, etc.) mais il est exclu par le modele V8.
    load_lag_1 et load_lag_4 ne sont plus crees (aucune dependance).
    """
    # Lags sur la charge
    for shift, name in [
        (96, "96"),       # J (exclu par V8, mais necessaire pour derivees)
        (192, "192"),     # J-1 meme heure (propre)
        (288, "288"),     # J-2 meme heure (propre)
        (672, "672"),     # J-7 (propre)
        (1344, "1344"),   # J-14 (propre)
        (2016, "2016"),   # J-21 (propre)
    ]:
        df = df.with_columns(
            pl.col("load").shift(shift).alias(f"load_lag_{name}")
        )

    # Lags sur la production PV
    for shift, name in [
        (96, "96"),       # J (exclu par V8)
        (192, "192"),     # J-1 (propre)
        (288, "288"),     # J-2 (propre)
        (672, "672"),     # J-7 (propre)
        (1344, "1344"),   # J-14 (propre)
    ]:
        df = df.with_columns(
            pl.col("pv_total").shift(shift).alias(f"pv_total_lag_{name}")
        )

    # Moyenne hebdomadaire 3 semaines (J-7, J-14, J-21)
    df = df.with_columns(
        ((pl.col("load_lag_672") + pl.col("load_lag_1344") + pl.col("load_lag_2016")) / 3.0)
        .alias("load_weekly_mean_3w")
    )

    # Ecart-type hebdomadaire 3 semaines
    lag_672 = pl.col("load_lag_672")
    lag_1344 = pl.col("load_lag_1344")
    lag_2016 = pl.col("load_lag_2016")
    mean_val = (lag_672 + lag_1344 + lag_2016) / 3.0
    mean_sq = (lag_672**2 + lag_1344**2 + lag_2016**2) / 3.0
    df = df.with_columns(
        (mean_sq - mean_val**2).clip(lower_bound=0.0).sqrt()
        .alias("load_weekly_std_3w")
    )

    # Moyenne PV 2 semaines (J-7, J-14)
    df = df.with_columns(
        ((pl.col("pv_total_lag_672") + pl.col("pv_total_lag_1344")) / 2.0)
        .alias("pv_weekly_mean_2w")
    )

    # Moyenne charge recente J-1 et J-2
    df = df.with_columns(
        ((pl.col("load_lag_192") + pl.col("load_lag_288")) / 2.0)
        .alias("load_recent_mean_2d")
    )

    # Ratio charge recente vs moyenne hebdomadaire
    df = df.with_columns(
        pl.when(pl.col("load_weekly_mean_3w").abs() > 0.001)
        .then(pl.col("load_lag_192") / pl.col("load_weekly_mean_3w"))
        .otherwise(1.0)
        .alias("load_ratio_recent_vs_weekly")
    )

    # Moyenne 4 semaines incluant J-1
    df = df.with_columns(
        ((pl.col("load_lag_192") + pl.col("load_lag_672")
          + pl.col("load_lag_1344") + pl.col("load_lag_2016")) / 4.0)
        .alias("load_weekly_mean_4w_inc_recent")
    )

    # Delta J-1 vs J-7
    df = df.with_columns(
        (pl.col("load_lag_192") - pl.col("load_lag_672"))
        .alias("load_delta_2d_vs_7d")
    )

    return df


# ============================================================
# 3. Position solaire + clearness index
# ============================================================

def add_solar_features(df: pl.DataFrame) -> pl.DataFrame:
    try:
        from pvlib import solarposition, location as pvloc
        import pandas as pd

        timestamps_pd = df["timestamp"].to_pandas()
        if not isinstance(timestamps_pd, pd.DatetimeIndex):
            timestamps_pd = pd.DatetimeIndex(timestamps_pd)

        solar = solarposition.get_solarposition(
            timestamps_pd, 46.2333, 7.3592, 500, method="nrel_numpy"
        )
        elev = solar["elevation"].clip(lower=0).values
        df = df.with_columns(
            pl.Series("solar_elevation", elev, dtype=pl.Float64)
        )

        site = pvloc.Location(46.2333, 7.3592, altitude=500, tz="UTC")
        cs = site.get_clearsky(timestamps_pd, model="ineichen")
        ghi_clearsky = cs["ghi"].values.astype(np.float64)
        df = df.with_columns(
            pl.Series("ghi_clearsky", ghi_clearsky, dtype=pl.Float64)
        )

        if "pred_glob_ctrl" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("ghi_clearsky") > 10.0)
                .then(
                    (pl.col("pred_glob_ctrl") / pl.col("ghi_clearsky")).clip(0.0, 1.5)
                )
                .otherwise(0.0)
                .alias("clearness_index_pred")
            )
            print("    Clearness index proxy OK")

        print(f"    Position solaire + ciel clair OK ({len(solar)} timestamps)")

    except ImportError:
        print("    Warning: pvlib non installe, solar features = 0")
        df = df.with_columns([
            pl.lit(0.0).alias("solar_elevation"),
            pl.lit(0.0).alias("ghi_clearsky"),
        ])
        if "pred_glob_ctrl" in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("clearness_index_pred"))

    return df


# ============================================================
# 4. Tendance capacite PV
# ============================================================

def add_pv_trend(df: pl.DataFrame) -> pl.DataFrame:
    ts_min = df["timestamp"].min()
    total_sec = (df["timestamp"].max() - ts_min).total_seconds()
    df = df.with_columns(
        ((pl.col("timestamp") - ts_min).dt.total_seconds() / total_sec)
        .alias("pv_capacity_trend")
    )
    return df


# ============================================================
# 5. Features d'interaction (PRED x deterministe uniquement)
# ============================================================

def add_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Interactions basees uniquement sur des PRED et des features deterministes.

    Les interactions qui utilisaient des donnees reelles (pv_total,
    meteo_temperature_2m) ont ete supprimees.
    """
    # PRED temperature x encodage horaire
    if "pred_t_2m_ctrl" in df.columns:
        df = df.with_columns([
            (pl.col("pred_t_2m_ctrl") * pl.col("hour_sin"))
            .alias("pred_temp_x_hour_sin"),
            (pl.col("pred_t_2m_ctrl") * pl.col("hour_cos"))
            .alias("pred_temp_x_hour_cos"),
        ])

    # PRED rayonnement x elevation solaire
    if "pred_glob_ctrl" in df.columns:
        df = df.with_columns(
            (pl.col("pred_glob_ctrl") * pl.col("solar_elevation"))
            .alias("pred_glob_x_elevation")
        )

    # PRED incertitude rayonnement x elevation solaire
    if "pred_glob_stde" in df.columns:
        df = df.with_columns(
            (pl.col("pred_glob_stde") * pl.col("solar_elevation"))
            .alias("pred_glob_stde_x_elevation")
        )

    # Clearness index x tendance PV
    if "clearness_index_pred" in df.columns:
        df = df.with_columns(
            (pl.col("clearness_index_pred") * pl.col("pv_capacity_trend"))
            .alias("clearness_x_pv_trend")
        )

    # IQR des predictions (q90 - q10)
    for var in ["pred_t_2m", "pred_glob", "pred_dursun"]:
        q90 = f"{var}_q90"
        q10 = f"{var}_q10"
        if q90 in df.columns and q10 in df.columns:
            df = df.with_columns(
                (pl.col(q90) - pl.col(q10)).alias(f"{var}_iqr")
            )

    return df


# ============================================================
# 6. Deltas meteo PRED vs semaine precedente (reelle shiftee)
# ============================================================

def add_weather_deltas(df: pl.DataFrame) -> pl.DataFrame:
    """Delta entre la prevision PRED et la meteo reelle d'il y a 7 jours.

    meteo_temperature_2m.shift(672) = temperature reelle de J-6,
    toujours disponible a 11h le jour J. Pas de leak.
    """
    if "pred_t_2m_ctrl" in df.columns and "meteo_temperature_2m" in df.columns:
        df = df.with_columns(
            (pl.col("pred_t_2m_ctrl") - pl.col("meteo_temperature_2m").shift(672))
            .alias("delta_temp_pred_vs_week_ago")
        )

    if "pred_glob_ctrl" in df.columns and "meteo_global_radiation" in df.columns:
        df = df.with_columns(
            (pl.col("pred_glob_ctrl") - pl.col("meteo_global_radiation").shift(672))
            .alias("delta_glob_pred_vs_week_ago")
        )

    return df


# ============================================================
# Pipeline principal
# ============================================================

def build_features(df: pl.DataFrame) -> pl.DataFrame:
    print("  [1/6] Variables calendaires + feries + ponts...")
    df = add_calendar_features(df)

    print("  [2/6] Lag features (hebdomadaires + J-1/J-2)...")
    df = add_lag_features(df)

    print("  [3/6] Position solaire + clearness index...")
    df = add_solar_features(df)

    print("  [4/6] Tendance capacite PV...")
    df = add_pv_trend(df)

    print("  [5/6] Features d'interaction (PRED x deterministe)...")
    df = add_interaction_features(df)

    print("  [6/6] Deltas meteo vs semaine precedente...")
    df = add_weather_deltas(df)

    n_before = df.shape[0]
    df = df.drop_nulls()
    n_after = df.shape[0]
    print(f"\n  Lignes supprimees (NaN lags) : {n_before - n_after}")
    print(f"  Dataset final : {n_after} lignes, {df.shape[1]} colonnes")

    return df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE ENGINEERING v8 - Forecast J+1 no-leak")
    print("=" * 60)

    print("\nChargement du dataset fusionne...")
    df = pl.read_parquet(DATA_PROCESSED / "dataset_merged.parquet")
    print(f"  {df.shape[0]} lignes, {df.shape[1]} colonnes")

    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    stde_cols = [c for c in pred_cols if "stde" in c]
    print(f"  Colonnes PRED : {len(pred_cols)} | stde : {len(stde_cols)}")

    print("\nConstruction des features...")
    df_features = build_features(df)

    output_path = DATA_PROCESSED / "dataset_features.parquet"
    df_features.write_parquet(output_path)
    print(f"\n  Sauvegarde : {output_path}")

    cols = sorted(df_features.columns)
    print(f"\n  Colonnes totales : {len(cols)}")
    for col in cols:
        print(f"    {col}")

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING v8 TERMINE")
    print("=" * 60)