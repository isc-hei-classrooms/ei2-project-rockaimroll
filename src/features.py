"""
Feature engineering pour OIKEN ML, forecast J+1.

Convention de prediction:
  - Run du modele : 11:00 locale du jour J (= matin de J)
  - Cible        : load_residual(T) = load(T) - forecast_load(T)
                   pour T dans [J+1 00:00, J+1 23:45], soit 96 timestamps.
  - Disponible a J 11:00 :
      * load(t), pv_total(t), meteo_*(t) pour t <= J 10:45
      * forecast_load(t) pour tout t <= J+1 23:45 (baseline publiee a J)
      * pred_*(t) pour tout t couvert par run COSMO-E "00" du jour J
      * Calendaires(T) deterministes

Regle de no-leak appliquee:
  - Tous les lags sur load/pv/meteo : >= 192 pas (2 jours).
    Lag 96 (=24h) est INTERDIT car pour une cible T = J+1 23:45,
    lag 96 = J+1 23:45 - 24h = J 23:45, donnee non disponible a J 11:00.
  - forecast_load(T) et pred_*(T) sont utilises a leur valeur courante.
  - Aucune feature n'utilise une source future de la cible.

Cette regle est verifiee par validate_no_leak() en fin de pipeline.

Pourquoi cible = residual ?
  La baseline OIKEN absorbe une partie de la croissance du parc PV et
  des saisonnalites grossieres. Predire le residual rend la cible plus
  stationnaire, plus facile a apprendre, et reduit la sensibilite a
  l'extrapolation hors plage d'entrainement (point critique pour le PV
  qui croit dans le temps).

Non-stationnarite PV (3 niveaux d'attenuation):
  Niveau 1 (ce module): expose `pv_time_index_days` (compteur monotone
    absolu) et `pv_yield_proxy_*` (rendement PV/clearsky), qui captent
    la croissance du parc. Note: les modeles a base d'arbres (LightGBM,
    XGBoost) saturent en extrapolation sur ces variables.
  Niveau 2 (model.py - TODO): cible = residual, comme deja choisi.
  Niveau 3 (post-processing - TODO): coefficient correctif `c(t)` sur
    la prediction PV finale, calibre sur la pente de croissance
    observee sur l'ensemble train.

Sortie: data/processed/dataset_features.parquet
Utilisation: python -m src.features
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from src.config import (
    DATA_PROCESSED,
    DATA_REPORTS,
    TIMEZONE_OIKEN,
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

# Seuils thermiques pour HDD/CDD. Sources: ASHRAE et pratique
# climatologique. 18 standard ; on ajoute 15 pour capter le "vrai
# chauffage" en zone alpine ou les Valaisans chauffent moins que
# la moyenne suisse.
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

# Vacances scolaires Valais. Sources: calendrier scolaire vs.ch.
# A maintenir manuellement. Le dataset OIKEN s'arrete au 29.09.2025
# donc les vacances d'automne 2025 ne sont pas necessaires pour le
# train set ; a ajouter pour la prediction future en production.
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
    """True si d tombe dans un intervalle de vacances scolaires VS."""
    for start, end in VACANCES_SCOLAIRES_VS:
        if start <= d <= end:
            return True
    return False


def _build_holiday_dates(years: list[int]) -> set[date]:
    """
    Liste des jours feries valaisans pour les annees donnees.
    Set vide si la lib `holidays` est absente.
    """
    try:
        import holidays
        ch_vs = holidays.Switzerland(prov="VS", years=years)
        return set(ch_vs.keys())
    except ImportError:
        return set()


def _holiday_distance_lookup(
    holiday_dates: set[date], min_date: date, max_date: date
) -> dict[date, tuple[int, int]]:
    """
    Pour chaque date de la plage, calcule (days_since_last, days_to_next)
    holiday. Plafonne a 30 jours.
    """
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
# 0. Cible : load_residual
# ============================================================

def add_target_residual(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cree la cible `load_residual` = load - forecast_load.

    Le modele J+1 apprendra ce residual. La prediction finale en
    inference se reconstruit par : pred_load = pred_residual + forecast_load.
    """
    if "load" not in df.columns or "forecast_load" not in df.columns:
        raise ValueError("Colonnes 'load' et 'forecast_load' requises.")

    df = df.with_columns(
        (pl.col("load") - pl.col("forecast_load")).alias("load_residual")
    )
    return df


# ============================================================
# 1. Calendaires + jours feries + vacances
# ============================================================

def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Toutes les features calendaires en heure LOCALE Europe/Zurich.
    Les patterns de consommation suivent l'heure locale, pas l'UTC.
    """
    ts_local = pl.col("timestamp").dt.convert_time_zone(TIMEZONE_OIKEN)

    # Composants temporels bruts
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

    # Encodages cycliques. (hour + minute/60) donne une valeur continue
    # 0..23.75 pour la resolution 15min. /365.25 absorbe les bissextiles.
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

    # Booleens calendaires
    df = df.with_columns(
        (pl.col("cal_weekday") >= 6).cast(pl.Int8).alias("cal_is_weekend")
    )

    # Jours feries Valais (via lib `holidays`)
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
        df = df.with_columns(
            pl.lit(0, dtype=pl.Int8).alias("cal_is_holiday")
        )

    # Vacances scolaires Valais (dates hardcodees)
    df = df.with_columns(
        pl.col("_date").map_elements(
            _is_school_holiday, return_dtype=pl.Boolean
        ).cast(pl.Int8).alias("cal_is_school_holiday")
    )

    # Distance au prochain/precedent ferie (signal de "pont a venir")
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

    # Jours ponts (vendredi avant ferie OU lundi apres ferie, hors w-e)
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

    # Lendemain de repos (lundi ou apres-ferie)
    df = df.with_columns(
        pl.when(
            (pl.col("_hol_yest") == 1) | (pl.col("cal_weekday") == 1)
        ).then(1).otherwise(0)
        .cast(pl.Int8).alias("cal_is_day_after_rest")
    )

    df = df.drop(["_date", "_hol_yest", "_hol_tom"])
    return df


# ============================================================
# 2. Position solaire + ciel clair (pvlib)
# ============================================================

def add_solar_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Position du soleil (NREL) et irradiance theorique ciel clair
    (Ineichen) calculees pour Sion. Fallback a 0 si pvlib absent.
    """
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

    # Clearness index sur prevision de rayonnement (proxy nebulosite)
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
# 3. Lags (cible + variables explicatives)
# ============================================================

def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Lags strictement no-leak pour J+1 (>= LAG_J1 = 192 pas).
    """
    # Lags de la cible (load_residual)
    for lag, suffix in [
        (LAG_J1, "192"),
        (LAG_J2, "288"),
        (LAG_W1, "672"),
        (LAG_W2, "1344"),
        (LAG_W3, "2016"),
        (LAG_W4, "2688"),
    ]:
        df = df.with_columns(
            pl.col("load_residual").shift(lag).alias(f"lag_residual_{suffix}")
        )

    # Lags de load brut (J-2, J-7, J-14)
    for lag, suffix in [(LAG_J1, "192"), (LAG_W1, "672"), (LAG_W2, "1344")]:
        df = df.with_columns(
            pl.col("load").shift(lag).alias(f"lag_load_{suffix}")
        )

    # Lags de forecast_load (baseline OIKEN J-2, J-7)
    for lag, suffix in [(LAG_J1, "192"), (LAG_W1, "672")]:
        df = df.with_columns(
            pl.col("forecast_load").shift(lag).alias(f"lag_forecast_{suffix}")
        )

    # Lags de pv_total (J-2, J-7, J-14)
    for lag, suffix in [(LAG_J1, "192"), (LAG_W1, "672"), (LAG_W2, "1344")]:
        df = df.with_columns(
            pl.col("pv_total").shift(lag).alias(f"lag_pv_{suffix}")
        )

    # Lags de meteo reelles (J-7 pour comparer aux PRED)
    meteo_lag_cols = [
        "meteo_temperature_2m",
        "meteo_global_radiation",
        "meteo_humidity",
        "meteo_wind_speed",
    ]
    for col in meteo_lag_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).shift(LAG_W1).alias(f"lag_{col}_672")
            )

    # Lags du clearsky GHI (utilises dans pv_yield_proxy)
    if "sun_clearsky_ghi" in df.columns:
        df = df.with_columns([
            pl.col("sun_clearsky_ghi").shift(LAG_W1).alias("lag_clearsky_672"),
            pl.col("sun_clearsky_ghi").shift(LAG_W2).alias("lag_clearsky_1344"),
        ])

    return df


# ============================================================
# 4. Statistiques rolling sur lags
# ============================================================

def add_rolling_lag_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Stats sur les 3 lags hebdomadaires (J-7, J-14, J-21) : mean, std,
    min, max. Plus stable qu'une rolling window classique car on agrege
    "memes conditions" (meme heure, meme jour de semaine).
    """
    triple_residual = [
        pl.col("lag_residual_672"),
        pl.col("lag_residual_1344"),
        pl.col("lag_residual_2016"),
    ]
    df = df.with_columns(
        pl.mean_horizontal(triple_residual).alias("roll_residual_mean_3w"),
    )
    sq_diff = [(c - pl.col("roll_residual_mean_3w")) ** 2
               for c in triple_residual]
    df = df.with_columns(
        (pl.sum_horizontal(sq_diff) / 3.0).sqrt()
        .alias("roll_residual_std_3w")
    )
    df = df.with_columns([
        pl.min_horizontal(triple_residual).alias("roll_residual_min_3w"),
        pl.max_horizontal(triple_residual).alias("roll_residual_max_3w"),
    ])

    # Moyenne 4 semaines incluant J-2 (mix recent + saisonnier)
    quad_residual = [
        pl.col("lag_residual_192"),
        pl.col("lag_residual_672"),
        pl.col("lag_residual_1344"),
        pl.col("lag_residual_2016"),
    ]
    df = df.with_columns(
        pl.mean_horizontal(quad_residual)
        .alias("roll_residual_mean_4w_inc_recent")
    )

    # Moyenne load brute recente (informatif meme si decorrele du residual)
    df = df.with_columns(
        ((pl.col("lag_load_192") + pl.col("lag_load_672")) / 2.0)
        .alias("roll_load_mean_recent")
    )

    # PV : mean et MAD (robuste) sur 2 semaines
    pv_pair = [pl.col("lag_pv_672"), pl.col("lag_pv_1344")]
    df = df.with_columns(
        pl.mean_horizontal(pv_pair).alias("roll_pv_mean_2w")
    )
    df = df.with_columns(
        ((pl.col("lag_pv_672") - pl.col("roll_pv_mean_2w")).abs()
         + (pl.col("lag_pv_1344") - pl.col("roll_pv_mean_2w")).abs())
        .alias("roll_pv_mad_2w")
    )

    # Trend semaine 1 vs semaine 2 (residual)
    df = df.with_columns(
        (pl.col("lag_residual_672") - pl.col("lag_residual_1344"))
        .alias("roll_residual_trend_w1_w2")
    )

    return df


# ============================================================
# 5. Derivees temperature
# ============================================================

def add_temperature_derived(df: pl.DataFrame) -> pl.DataFrame:
    """
    HDD/CDD et indicateurs thermiques sur la prevision de temperature.
    """
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
# 6. Dynamiques PV (gestion non-stationnarite)
# ============================================================

def add_pv_dynamics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Features dediees a la croissance du parc PV.

    Le modele a base d'arbres saturera en extrapolation sur ces
    variables. Solution complementaire: post-processing avec un facteur
    correctif calibre sur la pente de croissance observee.
    """
    # Compteur monotone depuis 2022-10-01 (en jours decimaux)
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

    # Yield proxies : rendement PV / clearsky theorique. Croit avec le
    # parc, indep. (au 1er ordre) de l'ensoleillement reel.
    if "lag_pv_672" in df.columns and "lag_clearsky_672" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("lag_clearsky_672") > 50.0)
            .then(
                (pl.col("lag_pv_672") / pl.col("lag_clearsky_672"))
                .clip(0.0, 100.0)
            )
            .otherwise(0.0)
            .alias("pv_yield_proxy_w1")
        )

    if "lag_pv_1344" in df.columns and "lag_clearsky_1344" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("lag_clearsky_1344") > 50.0)
            .then(
                (pl.col("lag_pv_1344") / pl.col("lag_clearsky_1344"))
                .clip(0.0, 100.0)
            )
            .otherwise(0.0)
            .alias("pv_yield_proxy_w2")
        )

    # Estimation locale de la pente de croissance
    if ("pv_yield_proxy_w1" in df.columns
            and "pv_yield_proxy_w2" in df.columns):
        df = df.with_columns(
            (pl.col("pv_yield_proxy_w1") - pl.col("pv_yield_proxy_w2"))
            .alias("pv_growth_w1_w2")
        )

    return df


# ============================================================
# 7. Incertitudes COSMO-E (IQR)
# ============================================================

def add_pred_uncertainty(df: pl.DataFrame) -> pl.DataFrame:
    """
    IQR (q90 - q10) pour chaque variable PRED. Mesure la dispersion
    de l'ensemble COSMO-E. Les `_stde` sont deja disponibles.
    """
    pred_vars = [
        "pred_t_2m", "pred_glob", "pred_dursun", "pred_tot_prec",
        "pred_relhum_2m", "pred_ff_10m", "pred_dd_10m", "pred_ps",
    ]
    for v in pred_vars:
        q10, q90 = f"{v}_q10", f"{v}_q90"
        if q10 in df.columns and q90 in df.columns:
            df = df.with_columns(
                (pl.col(q90) - pl.col(q10)).alias(f"iqr_{v}")
            )
    return df


# ============================================================
# 8. Deltas PRED vs observations historiques
# ============================================================

def add_weather_deltas(df: pl.DataFrame) -> pl.DataFrame:
    """
    Differences entre la prevision PRED et l'observation reelle d'il y
    a une semaine. Capture les anomalies meteo prevues.
    Strictement no-leak : meteo_*.shift(672) = J-7.
    """
    pairs = [
        ("pred_t_2m_ctrl", "lag_meteo_temperature_2m_672", "delta_temp_pred_vs_w1"),
        ("pred_glob_ctrl", "lag_meteo_global_radiation_672", "delta_glob_pred_vs_w1"),
        ("pred_relhum_2m_ctrl", "lag_meteo_humidity_672", "delta_humidity_pred_vs_w1"),
    ]
    for pred_col, lag_col, out_col in pairs:
        if pred_col in df.columns and lag_col in df.columns:
            df = df.with_columns(
                (pl.col(pred_col) - pl.col(lag_col)).alias(out_col)
            )
    return df


# ============================================================
# 9. Interactions
# ============================================================

def add_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Croisements ayant un sens metier:
      - PRED temp x cycles horaires : effet thermique varie selon l'heure
      - PRED rayonnement x elevation solaire : produit physique du PV
      - HDD x weekend/holiday : chauffage residentiel monte le w-e
      - lag_residual x is_holiday : le pattern J-7 differe en ferie
    """
    pairs = []

    # PRED temp x cycles
    if "pred_t_2m_ctrl" in df.columns and "cyc_hour_sin" in df.columns:
        pairs += [
            ("pred_t_2m_ctrl", "cyc_hour_sin", "inter_predtemp_x_hsin"),
            ("pred_t_2m_ctrl", "cyc_hour_cos", "inter_predtemp_x_hcos"),
        ]

    # PRED rayonnement x elevation solaire
    if "pred_glob_ctrl" in df.columns and "sun_elevation" in df.columns:
        pairs.append(("pred_glob_ctrl", "sun_elevation",
                      "inter_predglob_x_sunelev"))

    # PRED stde rayonnement x elevation
    if "pred_glob_stde" in df.columns and "sun_elevation" in df.columns:
        pairs.append(("pred_glob_stde", "sun_elevation",
                      "inter_predglobstde_x_sunelev"))

    # Clearness x time_index : croissance PV moderee par ensoleillement
    if ("sun_clearness_pred" in df.columns
            and "pv_time_index_days" in df.columns):
        pairs.append(("sun_clearness_pred", "pv_time_index_days",
                      "inter_clearness_x_timeidx"))

    # HDD x weekend/holiday
    if "temp_hdd_18" in df.columns and "cal_is_weekend" in df.columns:
        pairs.append(("temp_hdd_18", "cal_is_weekend",
                      "inter_hdd_x_weekend"))
    if "temp_hdd_18" in df.columns and "cal_is_holiday" in df.columns:
        pairs.append(("temp_hdd_18", "cal_is_holiday",
                      "inter_hdd_x_holiday"))

    # PRED temp x school_holiday
    if ("pred_t_2m_ctrl" in df.columns
            and "cal_is_school_holiday" in df.columns):
        pairs.append(("pred_t_2m_ctrl", "cal_is_school_holiday",
                      "inter_predtemp_x_schoolhol"))

    # lag_residual J-7 x is_holiday
    if ("lag_residual_672" in df.columns
            and "cal_is_holiday" in df.columns):
        pairs.append(("lag_residual_672", "cal_is_holiday",
                      "inter_lagres672_x_holiday"))

    for col_a, col_b, out_col in pairs:
        df = df.with_columns(
            (pl.col(col_a) * pl.col(col_b)).alias(out_col)
        )

    return df


# ============================================================
# 10. Validation no-leak
# ============================================================

def validate_no_leak(df: pl.DataFrame) -> dict:
    """
    Sanity checks:
      - Toutes les colonnes 'lag_*' ont un suffixe numerique >= LAG_J1
      - 'load_residual' present
      - Aucun null calendaire (cal_/cyc_)
    """
    warnings_list = []

    for col in df.columns:
        if col.startswith("lag_"):
            parts = col.rsplit("_", 1)
            if parts[-1].isdigit():
                n = int(parts[-1])
                if n < LAG_J1:
                    warnings_list.append(
                        f"LEAK SUSPECT: '{col}' lag={n} (< {LAG_J1})"
                    )

    if "load_residual" not in df.columns:
        warnings_list.append("Cible 'load_residual' absente")

    cal_cols = [c for c in df.columns if c.startswith(("cal_", "cyc_"))]
    for col in cal_cols:
        nc = df[col].null_count()
        if nc > 0:
            warnings_list.append(
                f"Calendaire '{col}' a {nc} nulls (devrait etre 0)"
            )

    return {"n_warnings": len(warnings_list), "warnings": warnings_list}


# ============================================================
# 11. Pipeline principal
# ============================================================

def build_features(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    Pipeline complet. Ordre important: solar avant lag_clearsky,
    calendar avant interactions qui utilisent cyc_*.
    """
    stats = {"steps": []}

    print("  [0/9] Cible : load_residual")
    df = add_target_residual(df)
    stats["steps"].append({"step": "target_residual",
                           "n_cols": len(df.columns)})

    print("  [1/9] Calendaires + feries + vacances + ponts")
    df = add_calendar_features(df)
    stats["steps"].append({"step": "calendar",
                           "n_cols": len(df.columns)})

    print("  [2/9] Position solaire + ciel clair (pvlib)")
    df = add_solar_features(df)
    stats["steps"].append({"step": "solar",
                           "n_cols": len(df.columns)})

    print("  [3/9] Lags (residual + load + pv + meteo + forecast)")
    df = add_lag_features(df)
    stats["steps"].append({"step": "lags",
                           "n_cols": len(df.columns)})

    print("  [4/9] Stats rolling sur lags hebdomadaires")
    df = add_rolling_lag_stats(df)
    stats["steps"].append({"step": "rolling",
                           "n_cols": len(df.columns)})

    print("  [5/9] Derivees temperature (HDD, CDD, gel)")
    df = add_temperature_derived(df)
    stats["steps"].append({"step": "temperature",
                           "n_cols": len(df.columns)})

    print("  [6/9] Dynamiques PV (non-stationnarite)")
    df = add_pv_dynamics(df)
    stats["steps"].append({"step": "pv_dynamics",
                           "n_cols": len(df.columns)})

    print("  [7/9] Incertitudes COSMO-E (IQR)")
    df = add_pred_uncertainty(df)
    stats["steps"].append({"step": "pred_uncertainty",
                           "n_cols": len(df.columns)})

    print("  [8/9] Deltas PRED vs meteo J-7")
    df = add_weather_deltas(df)
    stats["steps"].append({"step": "deltas",
                           "n_cols": len(df.columns)})

    print("  [9/9] Interactions")
    df = add_interactions(df)
    stats["steps"].append({"step": "interactions",
                           "n_cols": len(df.columns)})

    # Validation
    print("\n  Validation no-leak...")
    leak_check = validate_no_leak(df)
    stats["leak_check"] = leak_check
    if leak_check["n_warnings"] > 0:
        print(f"    [WARN] {leak_check['n_warnings']} warnings:")
        for w in leak_check["warnings"][:5]:
            print(f"      - {w}")
    else:
        print("    OK : aucun leak detecte")

    return df, stats


# ============================================================
# 12. Main
# ============================================================

def main() -> int:
    print("=" * 60)
    print("FEATURE ENGINEERING - OIKEN ML J+1 (residual baseline)")
    print("=" * 60)

    in_path = DATA_PROCESSED / "dataset_normalized.parquet"
    if not in_path.exists():
        print(f"\nERREUR: dataset normalise absent : {in_path}")
        print("Lance d'abord : python -m src.normalization")
        return 1

    print(f"\nChargement : {in_path}")
    df = pl.read_parquet(in_path)
    print(f"  Entree : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    print("\nConstruction des features...")
    df_feat, stats = build_features(df)

    # Drop des lignes incompletes (lags). On ne drop que sur les lag_/roll_
    # et load_residual (lui-meme nul si forecast_load est nul).
    print("\n  Drop des lignes avec lags incomplets...")
    n_before = df_feat.height
    drop_cols = [c for c in df_feat.columns
                 if c.startswith(("lag_", "roll_"))
                 or c == "load_residual"]
    df_feat = df_feat.drop_nulls(subset=drop_cols)
    n_after = df_feat.height
    print(f"    Lignes : {n_before:,} -> {n_after:,} "
          f"(perdues : {n_before - n_after:,})")

    # Resume par categorie
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

    target = df_feat["load_residual"]
    print(f"\n  Cible 'load_residual' :")
    print(f"    mean = {target.mean():.4f}, std = {target.std():.4f}")
    print(f"    min  = {target.min():.4f}, max = {target.max():.4f}")
    print(f"    nulls= {target.null_count()}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "dataset_features.parquet"
    df_feat.write_parquet(out_path)
    print(f"\n  -> {out_path}")

    DATA_REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "n_rows_in": n_before,
        "n_rows_out": n_after,
        "n_rows_dropped": n_before - n_after,
        "n_columns": df_feat.shape[1],
        "feature_categories": cat_counts,
        "target": {
            "name": "load_residual",
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
        },
        "no_leak_check": stats["leak_check"],
    }
    out_report = DATA_REPORTS / "features_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_report}")

    print("\nFEATURE ENGINEERING TERMINE")
    return 0


if __name__ == "__main__":
    sys.exit(main())