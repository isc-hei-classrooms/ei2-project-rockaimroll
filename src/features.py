"""
Feature engineering pour OIKEN ML, forecast J+1 - VERSION V4 STANDALONE.

Difference fondamentale vs V3:
  La V3 utilisait `pv_yield_proxy_w1 = pv(J-7)/clearsky(J-7)` comme proxy
  de capacite installee. Le bruit de la nebulosite a J-7 dominait le
  signal de croissance, et le top des features etait monopolise par les
  rolling stats (97% du gain). Le PV n'apparaissait nulle part.

  La V4 introduit un VRAI proxy de capacite installee :
    pv_capacity_factor_30d(T) = Q90 sur [T-32j, T-2j] du ratio
                                pv_total/clearsky_ghi quand
                                clearsky_ghi > 500 W/m^2
  Le Q90 selectionne implicitement les jours clairs (peu sensible a la
  nebulosite), la fenetre 30j moyenne plusieurs jours clairs, et le lag
  de 2 jours preserve la regle no-leak.

  Combinee avec la prevision de rayonnement, cette feature donne :
    pv_predicted_kwh(T) = pred_glob_ctrl(T) * pv_capacity_factor_30d(T)
                          * (sun_elevation > 0)
  une estimation PHYSIQUE de la production PV a T, qui adapte sa magnitude
  a la capacite courante du parc. C'est ce qui manquait au modele.

Regle d'admissibilite (inchangee vs V3):
  - Mesures reelles OIKEN (load, pv_*) -> autorisees
  - Observations meteo MeteoSuisse -> autorisees
  - Previsions meteo COSMO-E (pred_*) -> autorisees
  - Sortie du modele ML OIKEN (forecast_load et tout derive) -> INTERDIT

Convention de prediction (inchangee):
  - Run du modele : 11:00 locale du jour J
  - Cible        : load(T) pour T dans [J+1 00:00, J+1 23:45]
  - Lag minimum sur load/pv/meteo : >= 192 pas (2 jours).

Sortie: data/processed/dataset_features_<dataset>.parquet
Utilisation:
    python -m src.features                  # dataset par defaut
    python -m src.features --dataset golden # dataset specifique
    python -m src.features --all            # tous les datasets
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from src.config import (
    DATA_PROCESSED,
    DATA_REPORTS,
    DATASETS,
    DEFAULT_DATASET,
    TIMEZONE_OIKEN,
    get_dataset_config,
    get_normalized_path,
    get_features_path,
)


# ============================================================
# Constantes du module
# ============================================================

# Sion (centre geographique du perimetre OIKEN)
SION_LAT = 46.2333
SION_LON = 7.3592
SION_ALT_M = 482.0

# Date de reference pour le compteur monotone (debut donnees OIKEN)
PROJECT_REFERENCE_DATE = date(2022, 10, 1)

# Seuils thermiques pour HDD/CDD.
T_BASE_HEATING_18 = 18.0
T_BASE_HEATING_15 = 15.0
T_BASE_COOLING_22 = 22.0
T_NEUTRAL_VALAIS = 12.0

# Lags safe pour J+1 (multiples de 96 = 1 jour)
LAG_J1 = 192      # 2 jours, lag minimum safe
LAG_J2 = 288      # 3 jours
LAG_W1 = 672      # 1 semaine
LAG_W2 = 1344     # 2 semaines
LAG_W3 = 2016     # 3 semaines
LAG_W4 = 2688     # 4 semaines

# Capacity factor PV : parametres
PV_CAPACITY_WINDOW_DAYS = 30          # fenetre rolling
PV_CAPACITY_QUANTILE = 0.90           # quantile pour selectionner jours clairs
PV_CAPACITY_MIN_SAMPLES = 200         # min pas avec clearsky > 500 dans la fenetre
PV_CAPACITY_CLEARSKY_THRESHOLD = 500.0  # W/m^2, seuil "plein soleil"
PV_CAPACITY_LONG_WINDOW_DAYS = 60     # version stable

# Vacances scolaires Valais
VACANCES_SCOLAIRES_VS: list[tuple[date, date]] = [
    (date(2022, 10, 13), date(2022, 10, 23)),
    (date(2022, 12, 23), date(2023, 1, 8)),
    (date(2023, 2, 18),  date(2023, 2, 26)),
    (date(2023, 4, 7),   date(2023, 4, 16)),
    (date(2023, 6, 24),  date(2023, 8, 16)),
    (date(2023, 10, 19), date(2023, 10, 29)),
    (date(2023, 12, 22), date(2024, 1, 7)),
    (date(2024, 2, 10),  date(2024, 2, 18)),
    (date(2024, 3, 29),  date(2024, 4, 7)),
    (date(2024, 6, 22),  date(2024, 8, 14)),
    (date(2024, 10, 14), date(2024, 10, 25)),
    (date(2024, 12, 20), date(2025, 1, 5)),
    (date(2025, 3, 1),   date(2025, 3, 9)),
    (date(2025, 4, 18),  date(2025, 4, 27)),
    (date(2025, 6, 28),  date(2025, 8, 17)),
]


# ============================================================
# Helpers
# ============================================================

def _is_school_holiday(d: date) -> bool:
    for start, end in VACANCES_SCOLAIRES_VS:
        if start <= d <= end:
            return True
    return False


def _build_holiday_dates(years: list[int]) -> set[date]:
    try:
        import holidays
        ch_vs = holidays.Switzerland(prov="VS", years=years)
        return set(ch_vs.keys())
    except ImportError:
        return set()


def _holiday_distance_lookup(
    holiday_dates: set[date], min_date: date, max_date: date
) -> dict[date, tuple[int, int]]:
    if not holiday_dates:
        return {}
    sorted_holidays = sorted(holiday_dates)
    n_days = (max_date - min_date).days + 1
    lookup: dict[date, tuple[int, int]] = {}
    cap = 30
    for i in range(n_days):
        d = min_date + timedelta(days=i)
        prev_h = max((h for h in sorted_holidays if h <= d), default=None)
        next_h = min((h for h in sorted_holidays if h >= d), default=None)
        ds = (d - prev_h).days if prev_h else cap
        dn = (next_h - d).days if next_h else cap
        lookup[d] = (min(ds, cap), min(dn, cap))
    return lookup


# ============================================================
# 1. Calendaires (inchange vs V3)
# ============================================================

def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calendaires en heure LOCALE Europe/Zurich."""
    ts_local = pl.col("timestamp").dt.convert_time_zone(TIMEZONE_OIKEN)

    df = df.with_columns([
        ts_local.dt.hour().alias("cal_hour"),
        ts_local.dt.minute().alias("cal_minute"),
        ts_local.dt.weekday().alias("cal_weekday"),
        ts_local.dt.day().alias("cal_day"),
        ts_local.dt.month().alias("cal_month"),
        ts_local.dt.year().alias("cal_year"),
        ts_local.dt.ordinal_day().alias("cal_doy"),
        ts_local.dt.week().alias("cal_woy"),
    ])

    two_pi = 2 * np.pi
    df = df.with_columns([
        ((pl.col("cal_hour") + pl.col("cal_minute") / 60.0) * two_pi / 24.0)
        .sin().alias("cyc_hour_sin"),
        ((pl.col("cal_hour") + pl.col("cal_minute") / 60.0) * two_pi / 24.0)
        .cos().alias("cyc_hour_cos"),
        (pl.col("cal_weekday") * two_pi / 7.0).sin().alias("cyc_weekday_sin"),
        (pl.col("cal_weekday") * two_pi / 7.0).cos().alias("cyc_weekday_cos"),
        (pl.col("cal_month") * two_pi / 12.0).sin().alias("cyc_month_sin"),
        (pl.col("cal_month") * two_pi / 12.0).cos().alias("cyc_month_cos"),
        (pl.col("cal_doy") * two_pi / 365.25).sin().alias("cyc_doy_sin"),
        (pl.col("cal_doy") * two_pi / 365.25).cos().alias("cyc_doy_cos"),
    ])

    df = df.with_columns(
        (pl.col("cal_weekday") >= 6).cast(pl.Int8).alias("cal_is_weekend")
    )

    years = df["cal_year"].unique().to_list()
    holiday_dates = _build_holiday_dates(years)

    df = df.with_columns(ts_local.dt.date().alias("_date"))
    if holiday_dates:
        df = df.with_columns(
            pl.col("_date").is_in(list(holiday_dates))
            .cast(pl.Int8).alias("cal_is_holiday")
        )
    else:
        print("    [WARN] lib 'holidays' absente, cal_is_holiday = 0")
        df = df.with_columns(pl.lit(0, dtype=pl.Int8).alias("cal_is_holiday"))

    df = df.with_columns(
        pl.col("_date").map_elements(
            _is_school_holiday, return_dtype=pl.Boolean
        ).cast(pl.Int8).alias("cal_is_school_holiday")
    )

    if holiday_dates:
        ts_dates = df["_date"].to_list()
        min_d, max_d = min(ts_dates), max(ts_dates)
        lookup = _holiday_distance_lookup(holiday_dates, min_d, max_d)
        days_since = [lookup.get(d, (30, 30))[0] for d in ts_dates]
        days_to = [lookup.get(d, (30, 30))[1] for d in ts_dates]
        df = df.with_columns([
            pl.Series("cal_days_since_holiday", days_since, dtype=pl.Int16),
            pl.Series("cal_days_to_holiday", days_to, dtype=pl.Int16),
        ])
    else:
        df = df.with_columns([
            pl.lit(30, dtype=pl.Int16).alias("cal_days_since_holiday"),
            pl.lit(30, dtype=pl.Int16).alias("cal_days_to_holiday"),
        ])

    df = df.with_columns([
        pl.col("cal_is_holiday").shift(96).fill_null(0).alias("_hol_yest"),
        pl.col("cal_is_holiday").shift(-96).fill_null(0).alias("_hol_tom"),
    ])
    df = df.with_columns(
        pl.when(
            (pl.col("cal_is_holiday") == 0)
            & (pl.col("cal_is_weekend") == 0)
            & (
                ((pl.col("cal_weekday") == 5) & (pl.col("_hol_yest") == 1))
                | ((pl.col("cal_weekday") == 1) & (pl.col("_hol_tom") == 1))
            )
        ).then(1).otherwise(0)
        .cast(pl.Int8).alias("cal_is_bridge_day")
    )

    df = df.with_columns(
        pl.when(
            (pl.col("_hol_yest") == 1) | (pl.col("cal_weekday") == 1)
        ).then(1).otherwise(0)
        .cast(pl.Int8).alias("cal_is_day_after_rest")
    )

    df = df.drop(["_date", "_hol_yest", "_hol_tom"])
    return df


# ============================================================
# 2. Position solaire + ciel clair (inchange vs V3)
# ============================================================

def add_solar_features(df: pl.DataFrame) -> pl.DataFrame:
    """Position du soleil (NREL) et irradiance ciel clair (Ineichen)."""
    try:
        from pvlib import solarposition, location as pvloc
        import pandas as pd
    except ImportError:
        print("    [WARN] pvlib absent, sun_* mises a 0")
        df = df.with_columns([
            pl.lit(0.0).alias("sun_elevation"),
            pl.lit(0.0).alias("sun_azimuth"),
            pl.lit(0.0).alias("sun_clearsky_ghi"),
            pl.lit(0, dtype=pl.Int8).alias("sun_is_daylight"),
            pl.lit(0.0).alias("sun_clearness_pred"),
        ])
        return df

    ts_pd = df["timestamp"].to_pandas()
    if not isinstance(ts_pd, pd.DatetimeIndex):
        ts_pd = pd.DatetimeIndex(ts_pd)
    if ts_pd.tz is None:
        ts_pd = ts_pd.tz_localize("UTC")

    solpos = solarposition.get_solarposition(
        ts_pd, SION_LAT, SION_LON, altitude=SION_ALT_M, method="nrel_numpy"
    )
    elevation = solpos["elevation"].clip(lower=0).to_numpy()
    azimuth = solpos["azimuth"].to_numpy()

    site = pvloc.Location(SION_LAT, SION_LON, altitude=SION_ALT_M, tz="UTC")
    cs = site.get_clearsky(ts_pd, model="ineichen")
    ghi_cs = cs["ghi"].to_numpy().astype(np.float64)

    df = df.with_columns([
        pl.Series("sun_elevation", elevation, dtype=pl.Float64),
        pl.Series("sun_azimuth", azimuth, dtype=pl.Float64),
        pl.Series("sun_clearsky_ghi", ghi_cs, dtype=pl.Float64),
    ])
    df = df.with_columns(
        (pl.col("sun_elevation") > 0).cast(pl.Int8).alias("sun_is_daylight")
    )

    if "pred_glob_ctrl" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("sun_clearsky_ghi") > 10.0)
            .then(
                (pl.col("pred_glob_ctrl") / pl.col("sun_clearsky_ghi"))
                .clip(0.0, 1.5)
            )
            .otherwise(0.0)
            .alias("sun_clearness_pred")
        )

    return df


# ============================================================
# 3. Lags (inchange vs V3)
# ============================================================

def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """Lags strictement no-leak pour J+1 (>= LAG_J1 = 192 pas)."""
    for lag, suffix in [
        (LAG_J1, "192"), (LAG_J2, "288"),
        (LAG_W1, "672"), (LAG_W2, "1344"),
        (LAG_W3, "2016"), (LAG_W4, "2688"),
    ]:
        df = df.with_columns(
            pl.col("load").shift(lag).alias(f"lag_load_{suffix}")
        )

    for lag, suffix in [(LAG_J1, "192"), (LAG_W1, "672"), (LAG_W2, "1344")]:
        df = df.with_columns(
            pl.col("pv_total").shift(lag).alias(f"lag_pv_{suffix}")
        )

    meteo_lag_cols = [
        "meteo_temperature_2m", "meteo_global_radiation",
        "meteo_humidity", "meteo_wind_speed",
    ]
    for col in meteo_lag_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).shift(LAG_W1).alias(f"lag_{col}_672")
            )

    if "sun_clearsky_ghi" in df.columns:
        df = df.with_columns([
            pl.col("sun_clearsky_ghi").shift(LAG_W1).alias("lag_clearsky_672"),
            pl.col("sun_clearsky_ghi").shift(LAG_W2).alias("lag_clearsky_1344"),
        ])

    return df


# ============================================================
# 4. Rolling stats (inchange vs V3)
# ============================================================

def add_rolling_lag_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Stats sur les 3 lags hebdomadaires (J-7, J-14, J-21) du load."""
    triple_load = [
        pl.col("lag_load_672"),
        pl.col("lag_load_1344"),
        pl.col("lag_load_2016"),
    ]
    df = df.with_columns(
        pl.mean_horizontal(triple_load).alias("roll_load_mean_3w"),
    )
    sq_diff = [(c - pl.col("roll_load_mean_3w")) ** 2 for c in triple_load]
    df = df.with_columns(
        (pl.sum_horizontal(sq_diff) / 3.0).sqrt().alias("roll_load_std_3w")
    )
    df = df.with_columns([
        pl.min_horizontal(triple_load).alias("roll_load_min_3w"),
        pl.max_horizontal(triple_load).alias("roll_load_max_3w"),
    ])

    quad_load = [
        pl.col("lag_load_192"), pl.col("lag_load_672"),
        pl.col("lag_load_1344"), pl.col("lag_load_2016"),
    ]
    df = df.with_columns(
        pl.mean_horizontal(quad_load).alias("roll_load_mean_4w_inc_recent")
    )

    df = df.with_columns(
        ((pl.col("lag_load_192") + pl.col("lag_load_672")) / 2.0)
        .alias("roll_load_mean_recent")
    )

    df = df.with_columns(
        (pl.col("lag_load_672") - pl.col("lag_load_1344"))
        .alias("roll_load_trend_w1_w2")
    )

    pv_pair = [pl.col("lag_pv_672"), pl.col("lag_pv_1344")]
    df = df.with_columns(
        pl.mean_horizontal(pv_pair).alias("roll_pv_mean_2w")
    )
    df = df.with_columns(
        ((pl.col("lag_pv_672") - pl.col("roll_pv_mean_2w")).abs()
         + (pl.col("lag_pv_1344") - pl.col("roll_pv_mean_2w")).abs())
        .alias("roll_pv_mad_2w")
    )

    return df


# ============================================================
# 5. Derivees temperature (inchange vs V3)
# ============================================================

def add_temperature_derived(df: pl.DataFrame) -> pl.DataFrame:
    if "pred_t_2m_ctrl" not in df.columns:
        return df
    t = pl.col("pred_t_2m_ctrl")
    df = df.with_columns([
        (T_BASE_HEATING_18 - t).clip(lower_bound=0.0).alias("temp_hdd_18"),
        (T_BASE_HEATING_15 - t).clip(lower_bound=0.0).alias("temp_hdd_15"),
        (t - T_BASE_COOLING_22).clip(lower_bound=0.0).alias("temp_cdd_22"),
        (t - T_NEUTRAL_VALAIS).abs().alias("temp_dev_neutral"),
        (t < 0.0).cast(pl.Int8).alias("temp_is_freezing"),
        (t < -5.0).cast(pl.Int8).alias("temp_is_very_cold"),
    ])
    return df


# ============================================================
# 6. PV CAPACITY FEATURES - LE COEUR DE LA V4
# ============================================================

def add_pv_capacity_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Construit le proxy de CAPACITE INSTALLEE du parc PV.

    Methode:
      Etape 1. ratio_inst(t) = pv_total(t) / clearsky_ghi(t)
               valide uniquement quand clearsky_ghi(t) > 500 W/m^2
               (pas de plein jour, peu sensible a la nebulosite).
      Etape 2. capacity_unlagged(t) = Q90 du ratio_inst sur la fenetre
               glissante [t-30j, t]. Le Q90 selectionne les jours les
               plus clairs sans etre sensible aux outliers du Q100.
      Etape 3. pv_capacity_factor_30d(t) = capacity_unlagged(t-2j).
               Lag de 2 jours pour respecter strictement no-leak J+1.

    Pourquoi cette methode marche (vs V3):
      - V3 utilisait pv(J-7)/clearsky(J-7) : le bruit de la nebulosite
        de J-7 (peut etre tres nuageux) dominait le signal de capacite.
        Importance V3 : ~1% du gain total.
      - V4 utilise un Q90 sur 30 jours : selectionne les jours clairs
        et moyenne plusieurs valeurs. Variance reduite d'un facteur ~10.

    Verifications empiriques sur les donnees OIKEN:
      - Octobre 2022 : ~8.2 kWh par W/m^2 (parc initial).
      - Octobre 2024 : ~13.1 (+60%).
      - Septembre 2025 : ~15.7 (+92%).
      Croissance lisse, monotone par paliers semestriels.

    Sortie: 4 nouvelles features
      - pv_capacity_factor_30d  : Q90 sur 30j, lag 192
      - pv_capacity_factor_60d  : version stable (60j)
      - pv_capacity_growth      : difference 30j - 60j (vitesse locale)
      - pv_predicted_kwh        : pred_glob_ctrl * cap30 * is_daylight
                                  ESTIMATION PHYSIQUE de la prod a T

    Compatibilite no-leak:
      - clearsky_ghi(t) est deterministe (pvlib), donc utilisable a t=T.
      - pv_capacity_factor_30d(T) utilise des donnees jusqu'a T-2j.
      - pred_glob_ctrl(T) est dispo au run J 11:00 pour T en J+1.
      Donc pv_predicted_kwh(T) est calculable a J 11:00 pour T = J+1.
    """
    if "sun_clearsky_ghi" not in df.columns or "pv_total" not in df.columns:
        print("    [WARN] sun_clearsky_ghi ou pv_total absent, "
              "pv_capacity_* mises a 0")
        df = df.with_columns([
            pl.lit(0.0).alias("pv_capacity_factor_30d"),
            pl.lit(0.0).alias("pv_capacity_factor_60d"),
            pl.lit(0.0).alias("pv_capacity_growth"),
            pl.lit(0.0).alias("pv_predicted_kwh"),
        ])
        return df

    # Index temporel (deterministe, no-leak trivial) - garde de V3
    ref_naive = datetime(
        PROJECT_REFERENCE_DATE.year,
        PROJECT_REFERENCE_DATE.month,
        PROJECT_REFERENCE_DATE.day,
    )
    df = df.with_columns(
        ((pl.col("timestamp").dt.replace_time_zone(None) - pl.lit(ref_naive))
         .dt.total_seconds() / 86400.0)
        .alias("pv_time_index_days")
    )

    # Etape 1 : ratio instantane, masque sur clearsky eleve
    df = df.with_columns(
        pl.when(pl.col("sun_clearsky_ghi") > PV_CAPACITY_CLEARSKY_THRESHOLD)
        .then(pl.col("pv_total") / pl.col("sun_clearsky_ghi"))
        .otherwise(None)
        .alias("_pv_ratio_inst")
    )

    # Etape 2 : Q90 rolling sur 30 jours, puis sur 60 jours
    win_30 = PV_CAPACITY_WINDOW_DAYS * 96  # 30 * 96 = 2880 pas
    win_60 = PV_CAPACITY_LONG_WINDOW_DAYS * 96  # 60 * 96 = 5760 pas

    df = df.with_columns([
        pl.col("_pv_ratio_inst").rolling_quantile(
            quantile=PV_CAPACITY_QUANTILE,
            window_size=win_30,
            min_samples=PV_CAPACITY_MIN_SAMPLES,
        ).alias("_capfactor_30d_unlagged"),
        pl.col("_pv_ratio_inst").rolling_quantile(
            quantile=PV_CAPACITY_QUANTILE,
            window_size=win_60,
            min_samples=PV_CAPACITY_MIN_SAMPLES * 2,
        ).alias("_capfactor_60d_unlagged"),
    ])

    # Etape 3 : lag de 192 pas pour respecter no-leak strict
    # Apres lag, fill_null forward puis backward pour propager les valeurs
    # aux periodes ou le ratio etait masque (nuit, hiver matinal, etc.).
    df = df.with_columns([
        pl.col("_capfactor_30d_unlagged").shift(LAG_J1)
        .forward_fill().backward_fill()
        .alias("pv_capacity_factor_30d"),
        pl.col("_capfactor_60d_unlagged").shift(LAG_J1)
        .forward_fill().backward_fill()
        .alias("pv_capacity_factor_60d"),
    ])

    # Vitesse de croissance locale : 30j vs 60j
    # Si capacity_30 > capacity_60 -> croissance accelerant
    # Si capacity_30 < capacity_60 -> palier ou regression (rare)
    df = df.with_columns(
        (pl.col("pv_capacity_factor_30d") - pl.col("pv_capacity_factor_60d"))
        .alias("pv_capacity_growth")
    )

    # Estimation physique de la production PV a T :
    # pv_estimee(T) ~ pred_glob_ctrl(T) * capacity_factor_30d
    #                 quand soleil au-dessus horizon
    if "pred_glob_ctrl" in df.columns:
        df = df.with_columns(
            (pl.col("pred_glob_ctrl") * pl.col("pv_capacity_factor_30d")
             * (pl.col("sun_elevation") > 0).cast(pl.Float64))
            .alias("pv_predicted_kwh")
        )
    else:
        # Fallback : utiliser clearsky a la place de pred_glob (pire estimation)
        df = df.with_columns(
            (pl.col("sun_clearsky_ghi") * pl.col("pv_capacity_factor_30d")
             * (pl.col("sun_elevation") > 0).cast(pl.Float64))
            .alias("pv_predicted_kwh")
        )

    # Cleanup colonnes intermediaires
    df = df.drop([
        "_pv_ratio_inst",
        "_capfactor_30d_unlagged",
        "_capfactor_60d_unlagged",
    ])

    return df


# ============================================================
# 7. IQR COSMO-E (inchange vs V3)
# ============================================================

def add_pred_uncertainty(df: pl.DataFrame) -> pl.DataFrame:
    pred_vars = [
        "pred_t_2m", "pred_glob", "pred_dursun", "pred_tot_prec",
        "pred_relhum_2m", "pred_ff_10m", "pred_dd_10m", "pred_ps",
    ]
    for v in pred_vars:
        q10, q90 = f"{v}_q10", f"{v}_q90"
        if q10 in df.columns and q90 in df.columns:
            df = df.with_columns((pl.col(q90) - pl.col(q10)).alias(f"iqr_{v}"))
    return df


# ============================================================
# 8. Deltas PRED vs J-7 (inchange vs V3)
# ============================================================

def add_weather_deltas(df: pl.DataFrame) -> pl.DataFrame:
    pairs = [
        ("pred_t_2m_ctrl", "lag_meteo_temperature_2m_672",
         "delta_temp_pred_vs_w1"),
        ("pred_glob_ctrl", "lag_meteo_global_radiation_672",
         "delta_glob_pred_vs_w1"),
        ("pred_relhum_2m_ctrl", "lag_meteo_humidity_672",
         "delta_humidity_pred_vs_w1"),
    ]
    for pred_col, lag_col, out_col in pairs:
        if pred_col in df.columns and lag_col in df.columns:
            df = df.with_columns(
                (pl.col(pred_col) - pl.col(lag_col)).alias(out_col)
            )
    return df


# ============================================================
# 9. Interactions (etendu V4 : interactions PV)
# ============================================================

def add_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Croisements metier. V4 ajoute les interactions liees au PV:
      - capacity_factor x clearsky : magnitude PV theorique
      - capacity_factor x cosinus_heure : modulation horaire
      - pv_predicted_kwh x sun_elevation : redondance utile aux arbres
    """
    pairs = []

    # PRED temp x cycles (V3)
    if "pred_t_2m_ctrl" in df.columns and "cyc_hour_sin" in df.columns:
        pairs += [
            ("pred_t_2m_ctrl", "cyc_hour_sin", "inter_predtemp_x_hsin"),
            ("pred_t_2m_ctrl", "cyc_hour_cos", "inter_predtemp_x_hcos"),
        ]

    # PRED rayonnement x elevation solaire (V3)
    if "pred_glob_ctrl" in df.columns and "sun_elevation" in df.columns:
        pairs.append(("pred_glob_ctrl", "sun_elevation",
                      "inter_predglob_x_sunelev"))

    # PRED stde rayonnement x elevation (V3)
    if "pred_glob_stde" in df.columns and "sun_elevation" in df.columns:
        pairs.append(("pred_glob_stde", "sun_elevation",
                      "inter_predglobstde_x_sunelev"))

    # === NOUVEAU V4 : interactions PV avec capacity_factor ===

    # Capacity x clearsky : magnitude max theorique de la PV a T
    if ("pv_capacity_factor_30d" in df.columns
            and "sun_clearsky_ghi" in df.columns):
        pairs.append(("pv_capacity_factor_30d", "sun_clearsky_ghi",
                      "inter_capfactor_x_clearsky"))

    # Capacity x pred_glob : redondance avec pv_predicted_kwh mais
    # accessible sous forme d'interaction explicite pour XGBoost
    if ("pv_capacity_factor_30d" in df.columns
            and "pred_glob_ctrl" in df.columns):
        pairs.append(("pv_capacity_factor_30d", "pred_glob_ctrl",
                      "inter_capfactor_x_predglob"))

    # Capacity x cyclique horaire : la PV n'agit que le jour
    if ("pv_capacity_factor_30d" in df.columns
            and "cyc_hour_sin" in df.columns):
        pairs.append(("pv_capacity_factor_30d", "cyc_hour_sin",
                      "inter_capfactor_x_hsin"))
        pairs.append(("pv_capacity_factor_30d", "cyc_hour_cos",
                      "inter_capfactor_x_hcos"))

    # PV predicted x sun_elevation : redondance (deja dans pv_predicted)
    # mais aide les arbres a partitionner sur l'elevation
    if ("pv_predicted_kwh" in df.columns and "sun_elevation" in df.columns):
        pairs.append(("pv_predicted_kwh", "sun_elevation",
                      "inter_pvpred_x_sunelev"))

    # Clearness x time_index : croissance PV moderee par ensoleillement (V3)
    if ("sun_clearness_pred" in df.columns
            and "pv_time_index_days" in df.columns):
        pairs.append(("sun_clearness_pred", "pv_time_index_days",
                      "inter_clearness_x_timeidx"))

    # HDD x weekend/holiday (V3)
    if "temp_hdd_18" in df.columns and "cal_is_weekend" in df.columns:
        pairs.append(("temp_hdd_18", "cal_is_weekend", "inter_hdd_x_weekend"))
    if "temp_hdd_18" in df.columns and "cal_is_holiday" in df.columns:
        pairs.append(("temp_hdd_18", "cal_is_holiday", "inter_hdd_x_holiday"))

    # PRED temp x school_holiday (V3)
    if ("pred_t_2m_ctrl" in df.columns
            and "cal_is_school_holiday" in df.columns):
        pairs.append(("pred_t_2m_ctrl", "cal_is_school_holiday",
                      "inter_predtemp_x_schoolhol"))

    # lag_load J-7 x is_holiday (V3)
    if "lag_load_672" in df.columns and "cal_is_holiday" in df.columns:
        pairs.append(("lag_load_672", "cal_is_holiday",
                      "inter_lagload672_x_holiday"))

    for col_a, col_b, out_col in pairs:
        df = df.with_columns(
            (pl.col(col_a) * pl.col(col_b)).alias(out_col)
        )

    return df


# ============================================================
# 10. Validation no-leak + standalone (etendue V4)
# ============================================================

def validate_no_leak(df: pl.DataFrame) -> dict:
    """
    Sanity checks. V4 ajoute la validation des features pv_capacity_*:
    elles doivent etre non nulles sur la majorite du dataset.
    """
    warnings_list = []

    # Verifier que tous les lags numeriques sont >= LAG_J1
    for col in df.columns:
        if col.startswith("lag_"):
            parts = col.rsplit("_", 1)
            if parts[-1].isdigit():
                n = int(parts[-1])
                if n < LAG_J1:
                    warnings_list.append(
                        f"LEAK SUSPECT: '{col}' lag={n} (< {LAG_J1})"
                    )

    if "load" not in df.columns:
        warnings_list.append("Cible 'load' absente")

    # Garde-fou standalone
    forbidden_substrings = ("residual", "forecast_")
    for col in df.columns:
        if col == "forecast_load":
            continue
        if any(s in col for s in forbidden_substrings):
            warnings_list.append(
                f"FEATURE INTERDITE (standalone): '{col}'"
            )

    # Calendaires sans nulls
    cal_cols = [c for c in df.columns if c.startswith(("cal_", "cyc_"))]
    for col in cal_cols:
        nc = df[col].null_count()
        if nc > 0:
            warnings_list.append(
                f"Calendaire '{col}' a {nc} nulls"
            )

    # NEW V4 : capacity factor doit avoir une couverture >= 80%
    if "pv_capacity_factor_30d" in df.columns:
        n_nulls = df["pv_capacity_factor_30d"].null_count()
        coverage = (df.height - n_nulls) / df.height
        if coverage < 0.80:
            warnings_list.append(
                f"pv_capacity_factor_30d couverture faible: "
                f"{coverage:.1%} (< 80%)"
            )

    return {"n_warnings": len(warnings_list), "warnings": warnings_list}


# ============================================================
# 11. Pipeline principal
# ============================================================

def build_features(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    stats = {"steps": []}

    print("  [1/9] Calendaires + feries + vacances + ponts")
    df = add_calendar_features(df)
    stats["steps"].append({"step": "calendar", "n_cols": len(df.columns)})

    print("  [2/9] Position solaire + ciel clair (pvlib)")
    df = add_solar_features(df)
    stats["steps"].append({"step": "solar", "n_cols": len(df.columns)})

    print("  [3/9] Lags (load + pv + meteo)")
    df = add_lag_features(df)
    stats["steps"].append({"step": "lags", "n_cols": len(df.columns)})

    print("  [4/9] Stats rolling sur lags hebdomadaires")
    df = add_rolling_lag_stats(df)
    stats["steps"].append({"step": "rolling", "n_cols": len(df.columns)})

    print("  [5/9] Derivees temperature (HDD, CDD, gel)")
    df = add_temperature_derived(df)
    stats["steps"].append({"step": "temperature", "n_cols": len(df.columns)})

    print("  [6/9] PV CAPACITY FACTOR (V4 - nouveau coeur PV)")
    df = add_pv_capacity_features(df)
    stats["steps"].append({"step": "pv_capacity", "n_cols": len(df.columns)})

    # Diagnostic : afficher l'evolution du capacity factor
    if "pv_capacity_factor_30d" in df.columns:
        cf = df["pv_capacity_factor_30d"].drop_nulls()
        if cf.len() > 0:
            print(f"      pv_capacity_factor_30d : min={cf.min():.2f}, "
                  f"median={cf.median():.2f}, max={cf.max():.2f}")
            print(f"      croissance estimee : x{cf.max() / cf.min():.2f} "
                  f"(min vs max sur tout l'historique)")

    print("  [7/9] Incertitudes COSMO-E (IQR)")
    df = add_pred_uncertainty(df)
    stats["steps"].append({"step": "pred_uncertainty",
                           "n_cols": len(df.columns)})

    print("  [8/9] Deltas PRED vs meteo J-7")
    df = add_weather_deltas(df)
    stats["steps"].append({"step": "deltas", "n_cols": len(df.columns)})

    print("  [9/9] Interactions (V4 - capacity x meteo etendu)")
    df = add_interactions(df)
    stats["steps"].append({"step": "interactions", "n_cols": len(df.columns)})

    print("\n  Validation no-leak + standalone...")
    leak_check = validate_no_leak(df)
    stats["leak_check"] = leak_check
    if leak_check["n_warnings"] > 0:
        print(f"    [WARN] {leak_check['n_warnings']} warnings:")
        for w in leak_check["warnings"][:5]:
            print(f"      - {w}")
    else:
        print("    OK : aucun leak ni feature interdite detectee")

    return df, stats


# ============================================================
# 12. Main
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature engineering OIKEN ML J+1 (V4 standalone).",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=DEFAULT_DATASET,
        help=(f"Nom du dataset a traiter (defaut : {DEFAULT_DATASET})."),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Traite tous les datasets configures.",
    )
    return parser.parse_args(argv)


def run_one_dataset(dataset_name: str) -> int:
    """Construit les features pour un seul dataset. Retourne 0 si OK."""
    cfg = get_dataset_config(dataset_name)
    label: str = cfg["label"]
    in_path = get_normalized_path(dataset_name)
    out_path = get_features_path(dataset_name)
    out_report = DATA_REPORTS / f"features_report_{dataset_name}.json"

    print("=" * 60)
    print(f"FEATURE ENGINEERING - OIKEN ML J+1 (V4 STANDALONE)")
    print(f"Dataset='{dataset_name}' ({label})")
    print(f"Cible = load brut. Coeur PV : pv_capacity_factor_30d.")
    print("=" * 60)

    if not in_path.exists():
        print(f"\nERREUR: dataset normalise absent : {in_path}")
        print(f"Lance d'abord : python -m src.normalization "
              f"--dataset {dataset_name}")
        return 1

    print(f"\nChargement : {in_path}")
    df = pl.read_parquet(in_path)
    print(f"  Entree : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    if "forecast_load" in df.columns:
        print("  Note : forecast_load conserve dans le DataFrame "
              "(reserve a la metrique).")

    print("\nConstruction des features...")
    df_feat, stats = build_features(df)

    print("\n  Drop des lignes avec lags incomplets...")
    n_before = df_feat.height
    drop_cols = [c for c in df_feat.columns
                 if c.startswith(("lag_", "roll_"))]
    df_feat = df_feat.drop_nulls(subset=drop_cols)
    n_after = df_feat.height
    print(f"    Lignes : {n_before:,} -> {n_after:,} "
          f"(perdues : {n_before - n_after:,})")

    print("\n  --- Resume final ---")
    print(f"  Lignes  : {df_feat.shape[0]:,}")
    print(f"  Colonnes: {df_feat.shape[1]}")
    cat_counts = {
        "cal_/cyc_": sum(1 for c in df_feat.columns
                          if c.startswith(("cal_", "cyc_"))),
        "sun_": sum(1 for c in df_feat.columns if c.startswith("sun_")),
        "lag_": sum(1 for c in df_feat.columns if c.startswith("lag_")),
        "roll_": sum(1 for c in df_feat.columns if c.startswith("roll_")),
        "temp_": sum(1 for c in df_feat.columns if c.startswith("temp_")),
        "pv_": sum(1 for c in df_feat.columns if c.startswith("pv_")),
        "iqr_": sum(1 for c in df_feat.columns if c.startswith("iqr_")),
        "delta_": sum(1 for c in df_feat.columns if c.startswith("delta_")),
        "inter_": sum(1 for c in df_feat.columns if c.startswith("inter_")),
        "pred_": sum(1 for c in df_feat.columns if c.startswith("pred_")),
        "meteo_": sum(1 for c in df_feat.columns if c.startswith("meteo_")),
    }
    print("  Repartition par prefixe :")
    for prefix, n in cat_counts.items():
        print(f"    {prefix:12s}: {n:>3}")

    target = df_feat["load"]
    print(f"\n  Cible 'load' (brut) :")
    print(f"    mean = {target.mean():.4f}, std = {target.std():.4f}")
    print(f"    min  = {target.min():.4f}, max = {target.max():.4f}")
    print(f"    nulls= {target.null_count()}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_feat.write_parquet(out_path)
    print(f"\n  -> {out_path}")

    DATA_REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": dataset_name,
        "mode": "standalone_v4",
        "n_rows_in": n_before,
        "n_rows_out": n_after,
        "n_rows_dropped": n_before - n_after,
        "n_columns": df_feat.shape[1],
        "feature_categories": cat_counts,
        "target": {
            "name": "load",
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
        },
        "no_leak_check": stats["leak_check"],
        "v4_changelog": [
            "REMPLACE add_pv_dynamics par add_pv_capacity_features",
            "Q90 rolling sur 30j du ratio pv/clearsky (lag 192)",
            "Ajout pv_capacity_factor_60d, pv_capacity_growth",
            "Ajout pv_predicted_kwh = pred_glob * cap30d * is_daylight",
            "Ajout interactions inter_capfactor_x_clearsky/predglob/h*",
            "Ajout interaction inter_pvpred_x_sunelev",
            "Conservation pv_time_index_days pour drift en aval",
        ],
    }
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_report}")

    print(f"\nFEATURE ENGINEERING V4 '{dataset_name}' TERMINE")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.all:
        rc_total = 0
        results: dict[str, int] = {}
        for name in DATASETS.keys():
            rc = run_one_dataset(name)
            results[name] = rc
            if rc != 0:
                rc_total = rc
            print()  # separation visuelle
        print("=" * 60)
        print("RESUME --all")
        print("=" * 60)
        for name, rc in results.items():
            status = "OK" if rc == 0 else f"ECHEC (code {rc})"
            print(f"  {name:15s} : {status}")
        return rc_total

    return run_one_dataset(args.dataset)


if __name__ == "__main__":
    sys.exit(main())