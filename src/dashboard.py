"""
Dashboard HTML statique multi-modeles pour OIKEN ML.

Genere un fichier HTML autonome (Plotly.js depuis CDN) qui permet
d'explorer interactivement les predictions des modeles XGBoost
entraines sur differents datasets (original, golden). Le toggle
d'en-tete permet de basculer dynamiquement entre les modeles dans
la meme page.

Mode degrade : si l'un des modeles est absent sur disque, le pipeline
saute ce modele (warning console) et continue avec les autres. Si
aucun modele n'est disponible, le pipeline emet une erreur explicite.

Architecture du payload JSON :
  {
    "available_models": ["original", "golden"],
    "default_model": "original",
    "models": {
       "original": { meta, folds, daily, monthly, hourly, dow, by_day,
                     feature_importance, leak, drift_correction,
                     pv_correction_v4, model_params, tuning, cv_strategy },
       "golden":   { ... structure identique ... }
    },
    "constants": { ... }
  }

Sources (par dataset) :
  data/processed/predictions_xgboost_cv_<dataset>.parquet
  data/reports/model_xgboost_report_<dataset>.json
  data/processed/dataset_features_<dataset>.parquet  (irradiance, optionnel)

Sortie:
  data/reports/dashboard.html

Utilisation:
  python -m src.dashboard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl

try:
    from src.config import (
        PROJECT_ROOT,
        DATA_PROCESSED,
        DATA_REPORTS,
        DATASETS,
        DEFAULT_DATASET,
        get_features_path,
        get_predictions_path,
        get_model_report_path,
    )
except ImportError:
    # Fallback si lance hors du package (tests locaux)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_REPORTS = PROJECT_ROOT / "data" / "reports"
    DATASETS = {"original": {"label": "Original"}}
    DEFAULT_DATASET = "original"
    def get_features_path(d):
        return DATA_PROCESSED / f"dataset_features_{d}.parquet"
    def get_predictions_path(d):
        return DATA_PROCESSED / f"predictions_xgboost_cv_{d}.parquet"
    def get_model_report_path(d):
        return DATA_REPORTS / f"model_xgboost_report_{d}.json"


# ============================================================
# Configuration des modeles supportes
# ============================================================

MODEL_CONFIGS = {
    "original": {
        "display_name": "Original (3 ans)",
        "dataset": "original",
        "color_light": "#047857",   # vert profond
        "color_dark": "#34D399",    # vert lumineux
    },
    "golden": {
        "display_name": "Golden (2.6 ans)",
        "dataset": "golden",
        "color_light": "#7C3AED",   # violet profond
        "color_dark": "#A78BFA",    # violet lumineux
    },
}


# ============================================================
# Palette commune (CSS variables)
# ============================================================

PALETTE = {
    "real": "#0F172A",
    "baseline": "#1E40AF",
    "corrected": "#B45309",
    "error": "#991B1B",
    "muted": "#64748B",
    "accent": "#0891B2",
    "irradiance": "#D97706",
}

# Regle de no-leak (cf. features.py)
LAG_MIN_SAFE = 192   # = 2 jours en pas de 15 min

# Convention temporelle (en jours par rapport au run J 11:00)
# Conservees pour usage documentaire dans le bloc anti-leak (rule_summary)
# meme apres suppression de la frise temporelle.
RUN_TIME_DAYS = 0.0
TARGET_START_DAYS = 13.0 / 24.0
TARGET_END_DAYS = 13.0 / 24.0 + (23.0 + 45.0/60.0) / 24.0  # J+1 23:45 exact
LOAD_CUTOFF_DAYS = -9.0 / 24.0
METEO_REAL_CUTOFF_DAYS = 0.0


# ============================================================
# Libelles humains pour les features
# ============================================================

LAG_DAYS = {
    "192": "J-2", "288": "J-3",
    "672": "J-7", "1344": "J-14", "2016": "J-21", "2688": "J-28",
}

LAG_BASE_LABELS = {
    "residual": "Résidu (cible)",
    "load": "Charge",
    "forecast": "Prévision OIKEN",
    "pv": "Production PV",
    "clearsky": "Rayonnement ciel clair",
    "meteo_temperature_2m": "Température 2m",
    "meteo_global_radiation": "Rayonnement global",
    "meteo_humidity": "Humidité",
    "meteo_wind_speed": "Vitesse vent",
}

PRED_VAR_LABELS = {
    "t_2m": "Température 2m",
    "glob": "Rayonnement global",
    "dursun": "Durée d'ensoleillement",
    "tot_prec": "Précipitations",
    "relhum_2m": "Humidité relative",
    "ff_10m": "Vitesse vent 10m",
    "dd_10m": "Direction vent 10m",
    "ps": "Pression sol",
}

PRED_STAT_LABELS = {
    "ctrl": "centrale",
    "q10": "Q10",
    "q90": "Q90",
    "stde": "écart-type ens.",
}

FIXED_LABELS = {
    "forecast_load": "Prévision OIKEN (baseline)",
    "load": "Charge observée",
    "pv_total": "PV total observé",
    "meteo_temperature_2m": "Température observée (2m)",
    "meteo_global_radiation": "Rayonnement global observé",
    "meteo_humidity": "Humidité observée",
    "meteo_wind_speed": "Vent observé",

    "cal_hour": "Heure du jour",
    "cal_minute": "Minute",
    "cal_weekday": "Jour de la semaine (1-7)",
    "cal_day": "Jour du mois",
    "cal_month": "Mois",
    "cal_year": "Année",
    "cal_doy": "Jour de l'année",
    "cal_woy": "Semaine de l'année",
    "cal_is_weekend": "Week-end",
    "cal_is_holiday": "Jour férié",
    "cal_is_school_holiday": "Vacances scolaires",
    "cal_days_since_holiday": "Jours depuis dernier férié",
    "cal_days_to_holiday": "Jours avant prochain férié",
    "cal_is_bridge_day": "Jour de pont",
    "cal_is_day_after_rest": "Lendemain de jour férié",

    "cyc_hour_sin": "Heure (sinus)",
    "cyc_hour_cos": "Heure (cosinus)",
    "cyc_weekday_sin": "Jour semaine (sinus)",
    "cyc_weekday_cos": "Jour semaine (cosinus)",
    "cyc_month_sin": "Mois (sinus)",
    "cyc_month_cos": "Mois (cosinus)",
    "cyc_doy_sin": "Jour année (sinus)",
    "cyc_doy_cos": "Jour année (cosinus)",

    "sun_elevation": "Élévation du soleil",
    "sun_azimuth": "Azimut du soleil",
    "sun_clearsky_ghi": "Rayonnement ciel clair (GHI)",
    "sun_is_daylight": "Soleil au-dessus horizon",
    "sun_clearness_pred": "Indice de clarté (prévu / ciel clair)",

    "roll_residual_mean_3w": "Résidu, moyenne sur 3 semaines",
    "roll_residual_std_3w": "Résidu, écart-type sur 3 semaines",
    "roll_residual_min_3w": "Résidu, min sur 3 semaines",
    "roll_residual_max_3w": "Résidu, max sur 3 semaines",
    "roll_residual_mean_4w_inc_recent": "Résidu, moyenne 4 semaines (incl. J-2)",
    "roll_residual_trend_w1_w2": "Tendance résidu (J-7 vs J-14)",
    "roll_load_mean_recent": "Charge, moyenne récente (J-2 et J-7)",
    "roll_pv_mean_2w": "PV, moyenne sur 2 semaines",
    "roll_pv_mad_2w": "PV, déviation absolue 2 semaines",

    "temp_hdd_18": "Degrés-jours chauffage (base 18 °C)",
    "temp_hdd_15": "Degrés-jours chauffage (base 15 °C)",
    "temp_cdd_22": "Degrés-jours climatisation (base 22 °C)",
    "temp_dev_neutral": "Écart à température neutre (12 °C)",
    "temp_is_freezing": "Gel (T < 0 °C)",
    "temp_is_very_cold": "Très froid (T < -5 °C)",

    "pv_time_index_days": "Index temporel (jours depuis 2022-10-01)",
    "pv_yield_proxy_w1": "Rendement PV proxy (semaine -1)",
    "pv_yield_proxy_w2": "Rendement PV proxy (semaine -2)",
    "pv_growth_w1_w2": "Croissance rendement PV (S-1 vs S-2)",

    "delta_temp_pred_vs_w1": "Écart température prévue vs J-7",
    "delta_glob_pred_vs_w1": "Écart rayonnement prévu vs J-7",
    "delta_humidity_pred_vs_w1": "Écart humidité prévue vs J-7",

    "inter_predtemp_x_hsin": "Interaction : T° prévue × heure (sinus)",
    "inter_predtemp_x_hcos": "Interaction : T° prévue × heure (cosinus)",
    "inter_predglob_x_sunelev": "Interaction : rayonnement prévu × élévation soleil",
    "inter_predglobstde_x_sunelev": ("Interaction : incertitude rayonnement "
                                      "× élévation"),
    "inter_clearness_x_timeidx": ("Interaction : indice de clarté × "
                                   "tendance temporelle"),
    "inter_hdd_x_weekend": "Interaction : degrés-jours chauffage × week-end",
    "inter_hdd_x_holiday": "Interaction : degrés-jours chauffage × férié",
    "inter_predtemp_x_schoolhol": ("Interaction : T° prévue × "
                                    "vacances scolaires"),
    "inter_lagres672_x_holiday": "Interaction : résidu J-7 × jour férié",
    "inter_pvpred_x_sunelev": ("Interaction : PV prévu × élévation soleil "
                                "(amplitude effective)"),
    "inter_capfactor_x_hcos": ("Interaction : facteur de capacité PV "
                                "× heure (cosinus)"),
    "inter_capfactor_x_hsin": ("Interaction : facteur de capacité PV "
                                "× heure (sinus)"),
    "inter_capfactor_x_predglob": ("Interaction : facteur de capacité PV "
                                    "× rayonnement prévu"),
    "inter_capfactor_x_sunelev": ("Interaction : facteur de capacité PV "
                                   "× élévation soleil"),
}


# Helpers pour generer les libelles roll_<entite>_<stat>_<N>w generiquement
ROLL_ENTITY_LABELS = {
    "load": "Charge",
    "pv": "Production PV",
    "residual": "Résidu",
}
ROLL_STAT_LABELS = {
    "mean": "moyenne",
    "max": "maximum",
    "min": "minimum",
    "std": "écart-type",
    "mad": "déviation absolue médiane",
    "trend": "tendance",
}


def humanize_feature(name: str) -> str:
    """Libelle francais clair pour une feature (fallback : nom technique)."""
    if name in FIXED_LABELS:
        return FIXED_LABELS[name]

    if name.startswith("lag_"):
        body = name[4:]
        for base in sorted(LAG_BASE_LABELS, key=len, reverse=True):
            prefix = base + "_"
            if body.startswith(prefix):
                lag_str = body[len(prefix):]
                if lag_str.isdigit():
                    day_lbl = LAG_DAYS.get(lag_str, f"lag {lag_str}")
                    return f"{LAG_BASE_LABELS[base]} ({day_lbl})"

    if name.startswith("pred_"):
        rest = name[5:]
        for var_key in sorted(PRED_VAR_LABELS, key=len, reverse=True):
            prefix = var_key + "_"
            if rest.startswith(prefix):
                stat = rest[len(prefix):]
                var_lbl = PRED_VAR_LABELS[var_key]
                stat_lbl = PRED_STAT_LABELS.get(stat, stat)
                return f"Prévu : {var_lbl} ({stat_lbl})"

    if name.startswith("iqr_pred_"):
        var = name[len("iqr_pred_"):]
        var_lbl = PRED_VAR_LABELS.get(var, var)
        return f"Incertitude COSMO-E : {var_lbl} (Q90-Q10)"

    # Parser generique pour roll_<entite>_<stat>_<N>w[_inc_recent]
    # Exemples : roll_load_max_3w -> "Charge, maximum sur 3 semaines"
    #           roll_load_mean_4w_inc_recent -> "Charge, moyenne sur
    #           4 semaines (incluant J-2)"
    if name.startswith("roll_"):
        body = name[5:]
        inc_recent = body.endswith("_inc_recent")
        if inc_recent:
            body = body[:-len("_inc_recent")]
        # body attendu : "<entite>_<stat>_<N>w" (parts >= 3)
        parts = body.split("_")
        if len(parts) >= 3 and parts[-1].endswith("w"):
            window_token = parts[-1]
            n_weeks = window_token[:-1]
            stat = parts[-2]
            entity = "_".join(parts[:-2])
            ent_lbl = ROLL_ENTITY_LABELS.get(entity)
            stat_lbl = ROLL_STAT_LABELS.get(stat)
            if ent_lbl and stat_lbl and n_weeks.isdigit():
                base = f"{ent_lbl}, {stat_lbl} sur {n_weeks} semaines"
                if inc_recent:
                    base += " (incluant J-2)"
                return base

    return name


def categorize_feature(name: str) -> str:
    """Categorie d'une feature pour grouper visuellement."""
    if name.startswith(("cal_", "cyc_")):
        return "Calendaire"
    if name.startswith("sun_"):
        return "Solaire"
    if name.startswith("lag_"):
        return "Lags"
    if name.startswith("roll_"):
        return "Rolling stats"
    if name.startswith("temp_"):
        return "Température"
    if name.startswith("pv_"):
        return "PV / rendement"
    if name.startswith("iqr_"):
        return "Incertitude COSMO-E"
    if name.startswith("delta_"):
        return "Delta météo"
    if name.startswith("inter_"):
        return "Interactions"
    if name.startswith("pred_"):
        return "Prévision météo"
    if name == "forecast_load":
        return "Baseline OIKEN"
    return "Autre"


# ============================================================
# Audit anti-leak feature par feature
# ============================================================

def audit_feature_timing(name: str) -> dict:
    """
    Pour une feature, retourne :
      - source_label  : description courte de la donnee source
      - latest_days   : offset (j) de la donnee la PLUS RECENTE utilisee
      - status        : 'safe' | 'deterministic' | 'forecast'
                        | 'leak_suspect' | 'unknown'
      - lag_steps     : pour les lags, le nombre de pas 15 min
    """
    if name.startswith(("cal_", "cyc_")):
        return {"source_label": "Calculé depuis le timestamp",
                "latest_days": None, "status": "deterministic"}

    if name.startswith("sun_") and name not in ("sun_clearness_pred",):
        return {"source_label": "Calculé pvlib (position soleil)",
                "latest_days": None, "status": "deterministic"}

    if name == "sun_clearness_pred":
        return {"source_label": "Dérivé prévision COSMO-E",
                "latest_days": None, "status": "forecast"}

    if name == "forecast_load":
        return {"source_label": "Prévision OIKEN publiée au run",
                "latest_days": None, "status": "forecast"}

    if name.startswith("pred_") or name.startswith("iqr_pred_"):
        return {"source_label": "Run COSMO-E publié à J 00:00",
                "latest_days": None, "status": "forecast"}

    if name.startswith("temp_") or name.startswith("delta_"):
        return {"source_label": "Calculé depuis prévisions COSMO-E",
                "latest_days": None, "status": "forecast"}

    if name.startswith("lag_"):
        body = name[4:]
        parts = body.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            n = int(parts[1])
            latest = TARGET_END_DAYS - (n / 96.0)
            status = "leak_suspect" if (n < LAG_MIN_SAFE) else "safe"
            return {"source_label": f"Lag {n} pas (~{n/96:.1f} j)",
                    "latest_days": latest,
                    "status": status, "lag_steps": n}
        return {"source_label": "Lag (suffixe non parsé)",
                "latest_days": None, "status": "unknown"}

    if name.startswith("roll_"):
        if "inc_recent" in name or name == "roll_load_mean_recent":
            latest = TARGET_END_DAYS - (192 / 96.0)
            return {"source_label": "Stats sur lags (incl. J-2)",
                    "latest_days": latest, "status": "safe"}
        if "pv" in name:
            return {"source_label": "Stats PV sur lags hebdomadaires",
                    "latest_days": TARGET_END_DAYS - (672 / 96.0),
                    "status": "safe"}
        if "trend" in name:
            return {"source_label": "Trend résidu J-7 vs J-14",
                    "latest_days": TARGET_END_DAYS - (672 / 96.0),
                    "status": "safe"}
        return {"source_label": "Stats sur lags hebdomadaires",
                "latest_days": TARGET_END_DAYS - (672 / 96.0),
                "status": "safe"}

    if name == "pv_time_index_days":
        return {"source_label": "Index temporel (déterministe)",
                "latest_days": None, "status": "deterministic"}
    if name.startswith("pv_"):
        return {"source_label": "Yield proxy basé sur lags PV (J-7, J-14)",
                "latest_days": TARGET_END_DAYS - (672 / 96.0),
                "status": "safe"}

    if name.startswith("inter_"):
        return {"source_label": "Composition de plusieurs sources",
                "latest_days": None, "status": "safe"}

    return {"source_label": "Source inconnue",
            "latest_days": None, "status": "unknown"}

# ============================================================
# Chargement (par modele)
# ============================================================

def load_irradiance_features(dataset_name: str) -> pl.DataFrame | None:
    """
    Charge le DataFrame des features avec uniquement les colonnes
    necessaires pour l'irradiance, depuis le dataset_features du dataset
    indique. Retourne None si indisponible.
    """
    features_path = get_features_path(dataset_name)
    if not features_path.exists():
        return None
    try:
        feats = pl.read_parquet(
            features_path,
            columns=["timestamp", "meteo_global_radiation",
                     "sun_clearsky_ghi"],
        )
        if (feats["timestamp"].dtype == pl.Datetime
                and feats["timestamp"].dtype.time_zone is None):
            feats = feats.with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC")
            )
        return feats
    except Exception as e:
        print(f"  [WARN] Impossible de charger l'irradiance pour "
              f"'{dataset_name}' : {e}")
        return None


def load_model_data(
    model_key: str,
    feats_irr: pl.DataFrame | None,
) -> dict | None:
    """
    Charge predictions + rapport pour un modele donne.
    Retourne None si les fichiers sont absents.
    """
    cfg = MODEL_CONFIGS[model_key]
    dataset_name = cfg["dataset"]
    pred_path = get_predictions_path(dataset_name)
    report_path = get_model_report_path(dataset_name)

    if not pred_path.exists() or not report_path.exists():
        missing = []
        if not pred_path.exists():
            missing.append(str(pred_path))
        if not report_path.exists():
            missing.append(str(report_path))
        print(f"  [SKIP] {cfg['display_name']} : "
              f"fichier(s) manquant(s) -> {', '.join(missing)}")
        return None

    df = pl.read_parquet(pred_path)
    print(f"  [OK]   {cfg['display_name']:18s} : "
          f"{df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    if df["timestamp"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime(time_zone="UTC")
        )
    elif (df["timestamp"].dtype == pl.Datetime
          and df["timestamp"].dtype.time_zone is None):
        df = df.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC")
        )

    irradiance_source = None
    if feats_irr is not None:
        df = df.join(feats_irr, on="timestamp", how="left")
        if "meteo_global_radiation" in df.columns:
            obs_count = df["meteo_global_radiation"].drop_nulls().len()
            if obs_count > 0:
                irradiance_source = "obs"
                df = df.with_columns(
                    pl.col("meteo_global_radiation")
                    .alias("irradiance_w_m2")
                )
        if irradiance_source is None and "sun_clearsky_ghi" in df.columns:
            cs_count = df["sun_clearsky_ghi"].drop_nulls().len()
            if cs_count > 0:
                irradiance_source = "clearsky"
                df = df.with_columns(
                    pl.col("sun_clearsky_ghi").alias("irradiance_w_m2")
                )

    df = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Europe/Zurich")
        .alias("ts_local")
    )

    return {
        "model_key": model_key,
        "display_name": cfg["display_name"],
        "df": df,
        "report": report,
        "irradiance_source": irradiance_source,
    }


# ============================================================
# Agregations
# ============================================================

def compute_daily_aggregates(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns([
            pl.col("ts_local").dt.date().alias("date"),
            (pl.col("load_true") - pl.col("forecast_baseline")).abs()
                .alias("err_baseline"),
            (pl.col("load_true") - pl.col("load_pred_raw")).abs()
                .alias("err_model"),
            (pl.col("load_true") - pl.col("load_pred_corrected")).abs()
                .alias("err_corr"),
        ])
        .group_by("date", "fold")
        .agg([
            pl.col("err_baseline").mean().alias("mae_baseline"),
            pl.col("err_model").mean().alias("mae_model"),
            pl.col("err_corr").mean().alias("mae_corr"),
            pl.col("load_true").mean().alias("load_mean"),
            pl.col("load_true").max().alias("load_max"),
            pl.col("load_true").min().alias("load_min"),
            pl.len().alias("n_points"),
        ])
        .sort("date")
    )


def compute_monthly_aggregates(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns([
            pl.col("ts_local").dt.strftime("%Y-%m").alias("month"),
            (pl.col("load_true") - pl.col("forecast_baseline")).abs()
                .alias("err_baseline"),
            (pl.col("load_true") - pl.col("load_pred_raw")).abs()
                .alias("err_model"),
            (pl.col("load_true") - pl.col("load_pred_corrected")).abs()
                .alias("err_corr"),
        ])
        .group_by("month", "fold")
        .agg([
            pl.col("err_baseline").mean().alias("mae_baseline"),
            pl.col("err_model").mean().alias("mae_model"),
            pl.col("err_corr").mean().alias("mae_corr"),
            ((pl.col("err_model") - pl.col("err_baseline")) /
             pl.col("err_baseline") * 100).mean().alias("gain_pct"),
            pl.len().alias("n_points"),
        ])
        .sort("month")
    )


def compute_hourly_diagnostics(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns([
            pl.col("ts_local").dt.hour().alias("hour"),
            (pl.col("load_true") - pl.col("forecast_baseline")).abs()
                .alias("err_baseline"),
            (pl.col("load_true") - pl.col("load_pred_raw")).abs()
                .alias("err_model"),
        ])
        .group_by("hour", "fold")
        .agg([
            pl.col("err_baseline").mean().alias("mae_baseline"),
            pl.col("err_model").mean().alias("mae_model"),
        ])
        .sort("hour", "fold")
    )


def compute_dow_diagnostics(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns([
            pl.col("ts_local").dt.weekday().alias("dow"),
            (pl.col("load_true") - pl.col("forecast_baseline")).abs()
                .alias("err_baseline"),
            (pl.col("load_true") - pl.col("load_pred_raw")).abs()
                .alias("err_model"),
        ])
        .group_by("dow", "fold")
        .agg([
            pl.col("err_baseline").mean().alias("mae_baseline"),
            pl.col("err_model").mean().alias("mae_model"),
        ])
        .sort("dow", "fold")
    )


# ============================================================
# Serialisation
# ============================================================

def df_to_compact_json(df: pl.DataFrame) -> dict:
    out = {}
    for col in df.columns:
        s = df[col]
        dt = s.dtype
        if dt == pl.Date:
            out[col] = s.dt.strftime("%Y-%m-%d").to_list()
        elif dt == pl.Datetime:
            out[col] = s.dt.strftime("%Y-%m-%dT%H:%M").to_list()
        elif dt in (pl.Float32, pl.Float64):
            out[col] = [None if v is None else round(float(v), 5)
                        for v in s.to_list()]
        else:
            out[col] = s.to_list()
    return out


def get_model_params_from_report(report: dict) -> dict:
    """Recupere les hyperparams XGBoost depuis le rapport."""
    for key in ("xgb_params_used", "model_params"):
        if key in report:
            return report[key]
    return {}


def build_model_payload(model_data: dict) -> dict:
    """
    Construit le sous-payload pour un modele.
    """
    df = model_data["df"]
    report = model_data["report"]
    irradiance_source = model_data["irradiance_source"]
    display_name = model_data["display_name"]

    daily = compute_daily_aggregates(df)
    monthly = compute_monthly_aggregates(df)
    hourly = compute_hourly_diagnostics(df)
    dow = compute_dow_diagnostics(df)

    df_local = df.with_columns(
        pl.col("ts_local").dt.strftime("%Y-%m-%d").alias("date_key"),
        pl.col("ts_local").dt.strftime("%H:%M").alias("hhmm"),
    )
    has_irradiance = "irradiance_w_m2" in df_local.columns

    by_day = {}
    for date_key, sub in df_local.group_by("date_key", maintain_order=True):
        sub_sorted = sub.sort("ts_local")
        date_str = date_key[0]
        day_obj = {
            "hhmm": sub_sorted["hhmm"].to_list(),
            "load_true": [round(float(v), 4)
                          for v in sub_sorted["load_true"].to_list()],
            "forecast_baseline": [round(float(v), 4)
                                  for v in sub_sorted["forecast_baseline"]
                                  .to_list()],
            "load_pred_raw": [round(float(v), 4)
                              for v in sub_sorted["load_pred_raw"].to_list()],
            "load_pred_corrected": [round(float(v), 4)
                                    for v in sub_sorted["load_pred_corrected"]
                                    .to_list()],
            "fold": int(sub_sorted["fold"][0]),
            "n_points": sub_sorted.height,
        }
        if has_irradiance:
            irr_vals = sub_sorted["irradiance_w_m2"].to_list()
            day_obj["irradiance"] = [None if v is None else round(float(v), 1)
                                     for v in irr_vals]
        by_day[date_str] = day_obj

    fi_top30 = report.get("feature_importance_top30", [])
    fi_with_meta = []
    for name, score in fi_top30:
        audit = audit_feature_timing(name)
        fi_with_meta.append({
            "name": name,
            "label": humanize_feature(name),
            "score": float(score),
            "category": categorize_feature(name),
            "audit": audit,
        })

    leak_payload = build_leak_payload(report, fi_with_meta)
    model_params = get_model_params_from_report(report)

    return {
        "meta": {
            "model": report.get("model", display_name),
            "display_name": display_name,
            "version": report.get("version", "?"),
            "n_features": report.get("n_features", 0),
            "n_days": len(by_day),
            "n_predictions": int(df.height),
            "ts_min": df["ts_local"].min().strftime("%Y-%m-%d %H:%M"),
            "ts_max": df["ts_local"].max().strftime("%Y-%m-%d %H:%M"),
            "irradiance_source": irradiance_source,
            "irradiance_label": ("Rayonnement observé (W/m²)"
                                 if irradiance_source == "obs"
                                 else "Rayonnement ciel clair (W/m²)"
                                 if irradiance_source == "clearsky"
                                 else None),
        },
        "tuning": report.get("tuning"),
        "model_params": model_params,
        "cv_strategy": report.get("cv_strategy", {}),
        "drift_correction": report.get("drift_correction", {}),
        "residual_target": report.get("residual_target"),
        "pv_correction_v4": report.get("pv_correction"),
        "folds": report.get("folds", []),
        "daily": df_to_compact_json(daily),
        "monthly": df_to_compact_json(monthly),
        "hourly": df_to_compact_json(hourly),
        "dow": df_to_compact_json(dow),
        "by_day": by_day,
        "feature_importance": fi_with_meta,
        "leak": leak_payload,
    }


def build_leak_payload(report: dict, fi_with_meta: list) -> dict:
    """
    Construit la section anti-leak du payload (par modele).

    Note : la frise temporelle (categories_for_timeline) a ete supprimee
    car elle posait des problemes de mise en page (chevauchement des
    labels 'Run J 11:00', 'Cutoff load J 02:00', 'Cutoff modele
    J-1 23:45' dans une fenetre temporelle trop etroite). Le panel
    anti-leak conserve : status banner, regle textuelle, statut top 30
    (camembert), output validation pipeline, audit feature par feature.
    """
    leak_check = (report.get("no_leak_check")
                  or report.get("leak_check")
                  or {})
    n_warn = leak_check.get("n_warnings", 0)
    warnings_list = leak_check.get("warnings", [])

    status_counts = {"safe": 0, "deterministic": 0, "forecast": 0,
                     "leak_suspect": 0, "unknown": 0}
    for f in fi_with_meta:
        status = f["audit"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "validation_pipeline": {
            "n_warnings": n_warn,
            "warnings": warnings_list,
            "lag_min_steps": leak_check.get("lag_min_steps", LAG_MIN_SAFE),
        },
        "status_counts_top30": status_counts,
        "rule_summary": {
            "run_time": "J 11:00 (heure locale)",
            "target_horizon": ("J+1 00:00 → J+1 23:45 "
                               "(96 timestamps de 15 min)"),
            "lag_min_safe_steps": LAG_MIN_SAFE,
            "lag_min_safe_explanation": (
                "Pour cible T = J+1 23:45, lag k pas donne valeur à "
                "T - k×15min. Au moment du run (J 11:00), il est interdit "
                "d'utiliser des données plus récentes que J 10:45. Le "
                "pire cas est T - LAG, qui doit rester ≤ J 10:45 pour "
                "toute T dans [J+1 00:00, J+1 23:45]. Cela impose "
                f"LAG ≥ {LAG_MIN_SAFE} pas (= 2 jours)."
            ),
            "load_cutoff": (
                "Charge OIKEN : disponible avec une latence d'environ 9h. "
                "À J 11:00, dernière mesure réelle = J 02:00. "
                "Le modèle applique une marge supplémentaire (LAG ≥ 192) "
                "pour ramener la donnée la plus récente à J-1 23:45."
            ),
            "meteo_cutoff": (
                "Météo réelle (capteurs MeteoSuisse) : quasi temps réel. "
                "À J 11:00, dernière mesure ≈ J 11:00 elle-même. "
                "Mais la même règle LAG ≥ 192 ramène au même cutoff."
            ),
        },
    }


def build_combined_payload(loaded_models: dict) -> dict:
    """
    Combine les payloads de tous les modeles charges.
    """
    available = list(loaded_models.keys())
    default_model = available[0] if available else None

    model_meta = {}
    for key in available:
        cfg = MODEL_CONFIGS[key]
        model_meta[key] = {
            "key": key,
            "display_name": cfg["display_name"],
            "color_light": cfg["color_light"],
            "color_dark": cfg["color_dark"],
        }

    models_payloads = {}
    for key, data in loaded_models.items():
        models_payloads[key] = build_model_payload(data)

    return {
        "available_models": available,
        "default_model": default_model,
        "model_meta": model_meta,
        "models": models_payloads,
        "global_meta": {
            "generated_at": datetime.utcnow()
                .strftime("%Y-%m-%d %H:%M UTC"),
            "n_models_loaded": len(available),
        },
        "constants": {
            "lag_min_safe_steps": LAG_MIN_SAFE,
            "lag_min_safe_days": LAG_MIN_SAFE / 96.0,
        },
    }


# ============================================================
# Template HTML
# ============================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OIKEN ML, dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root {
  --bg: #FAFAF7;
  --bg-elev: #FFFFFF;
  --bg-sunken: #F1F1EC;
  --border: #E5E5DF;
  --border-strong: #C5C5BC;
  --text: #1A1A1A;
  --text-muted: #5C5C58;
  --text-faint: #8A8A85;
  --color-real: __COLOR_REAL__;
  --color-baseline: __COLOR_BASELINE__;
  --color-corrected: __COLOR_CORRECTED__;
  --color-error: __COLOR_ERROR__;
  --color-accent: __COLOR_ACCENT__;
  --color-irradiance: __COLOR_IRRADIANCE__;
  --color-model: __COLOR_MODEL_DEFAULT__;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.06);
  --radius: 6px;
}
[data-theme="dark"] {
  --bg: #0F1115;
  --bg-elev: #181B22;
  --bg-sunken: #0A0C10;
  --border: #2A2F3A;
  --border-strong: #3D4350;
  --text: #E8E8E5;
  --text-muted: #A0A4AB;
  --text-faint: #6B6F78;
  --color-real: #F5F5F2;
  --color-baseline: #60A5FA;
  --color-corrected: #FBBF24;
  --color-error: #F87171;
  --color-accent: #22D3EE;
  --color-irradiance: #FBBF24;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  background: var(--bg);
  color: var(--text);
  font-family: 'IBM Plex Sans', -apple-system, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.num, .mono {
  font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
h1, h2, h3 {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 600;
  letter-spacing: -0.01em;
}

/* ============ Header ============ */
.header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--bg-elev);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 24px;
}
.brand { display: flex; flex-direction: column; gap: 2px; }
.brand-title {
  font-family: 'Fraunces', serif;
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.brand-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-faint);
  font-variant-numeric: tabular-nums;
}
.header-stats {
  display: flex;
  gap: 18px;
  font-size: 12px;
  color: var(--text-muted);
}
.header-stat strong {
  font-family: 'JetBrains Mono', monospace;
  color: var(--text);
  margin-right: 4px;
}

/* ============ Switcher de modele ============ */
.model-switcher {
  margin-left: auto;
  display: flex;
  background: var(--bg-sunken);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px;
  gap: 2px;
}
.model-switcher.hidden { display: none; }
.model-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  padding: 6px 14px;
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  border-radius: 4px;
  transition: background 0.15s, color 0.15s;
  letter-spacing: 0.01em;
}
.model-btn:hover { color: var(--text); }
.model-btn.active {
  background: var(--color-model);
  color: white;
  box-shadow: var(--shadow-sm);
}
.model-btn .model-btn-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
  background: currentColor;
  opacity: 0.7;
}
.model-btn.active .model-btn-dot {
  background: white;
  opacity: 1;
}

.theme-toggle {
  background: transparent;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius);
  color: var(--text);
  padding: 6px 12px;
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}
.theme-toggle:hover { background: var(--bg-sunken); }

/* ============ Tabs ============ */
.tabs {
  display: flex;
  gap: 0;
  background: var(--bg-elev);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  overflow-x: auto;
}
.tab {
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--text-muted);
  padding: 12px 18px;
  font-family: inherit;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: color 0.15s, border-color 0.15s;
}
.tab:hover { color: var(--text); }
.tab.active {
  color: var(--text);
  border-bottom-color: var(--color-model);
}
.tab-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-faint);
  margin-right: 6px;
}

/* ============ Content ============ */
.content {
  padding: 24px;
  max-width: 1600px;
  margin: 0 auto;
}
.panel { display: none; }
.panel.active { display: block; }
.panel-title { font-size: 22px; margin-bottom: 4px; }
.panel-subtitle {
  color: var(--text-muted);
  font-size: 13px;
  margin-bottom: 20px;
}
.panel-subtitle .model-tag {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 10px;
  background: var(--color-model);
  color: white;
  font-size: 11px;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  margin: 0 2px;
}

/* ============ Cards ============ */
.cards { display: grid; gap: 16px; }
.cards.cols-2 { grid-template-columns: repeat(2, 1fr); }
.cards.cols-3 { grid-template-columns: repeat(3, 1fr); }
.cards.cols-4 { grid-template-columns: repeat(4, 1fr); }
.cards.cols-5 { grid-template-columns: repeat(5, 1fr); }
@media (max-width: 1100px) {
  .cards.cols-3, .cards.cols-4, .cards.cols-5 {
    grid-template-columns: repeat(2, 1fr);
  }
}
.card {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}
.card-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-faint);
  margin-bottom: 8px;
}
.kpi-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 28px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  margin-bottom: 4px;
}
.kpi-sub { font-size: 12px; color: var(--text-muted); }
.kpi-sub .delta-pos { color: var(--color-error); font-weight: 500; }
.kpi-sub .delta-neg { color: var(--color-model); font-weight: 500; }

.plot {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
  margin-bottom: 16px;
}
.plot-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin: 4px 4px 8px;
}
.plot-meta {
  font-size: 11px;
  color: var(--text-faint);
  font-family: 'JetBrains Mono', monospace;
  margin: 0 4px 8px;
}

.controls {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
  margin-bottom: 16px;
}
.controls-label { font-size: 12px; color: var(--text-muted); font-weight: 500; }
.control {
  background: var(--bg);
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  color: var(--text);
  padding: 6px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  cursor: pointer;
}
.control:focus { outline: 2px solid var(--color-accent); outline-offset: -1px; }
.control-group { display: flex; gap: 4px; align-items: center; }
.btn {
  background: var(--bg);
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  color: var(--text);
  padding: 5px 10px;
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}
.btn:hover { background: var(--bg-sunken); }
.btn.active {
  background: var(--color-accent);
  color: white;
  border-color: var(--color-accent);
}

table.tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
table.tbl th, table.tbl td {
  padding: 8px 10px; text-align: right;
  border-bottom: 1px solid var(--border);
}
table.tbl th {
  font-weight: 600; color: var(--text-muted);
  text-transform: uppercase; font-size: 10px;
  letter-spacing: 0.06em; text-align: right;
}
table.tbl th:first-child, table.tbl td:first-child { text-align: left; }
table.tbl td.num {
  font-family: 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums;
}
table.tbl tr:hover td { background: var(--bg-sunken); }
.tbl-wrap {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px 12px;
}

/* ============ Calendar heatmap ============ */
.calendar-wrap { display: grid; gap: 16px; }
.calendar-month {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px;
}
.cal-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
}
.cal-month-name {
  font-family: 'Fraunces', serif;
  font-size: 16px;
  font-weight: 600;
}
.cal-month-mae {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--text-muted);
}
.cal-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 3px;
}
.cal-dow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  color: var(--text-faint);
  text-align: center;
  padding: 3px 0;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.cal-day {
  aspect-ratio: 1;
  border-radius: 3px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: border-color 0.1s, transform 0.1s;
  position: relative;
}
.cal-day:hover {
  border-color: var(--color-accent);
  transform: scale(1.06);
  z-index: 2;
}
.cal-day.empty {
  background: transparent;
  cursor: default;
  pointer-events: none;
}
.cal-day-num { font-size: 11px; font-weight: 500; }
.cal-day-mae { font-size: 9px; opacity: 0.85; }

.cal-legend {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}
.cal-legend-bar {
  flex: 1;
  min-width: 200px;
  height: 12px;
  border-radius: 3px;
  background: linear-gradient(to right,
    hsla(120, 70%, 45%, 0.85),
    hsla(60, 70%, 50%, 0.85),
    hsla(0, 70%, 50%, 0.85));
}
.cal-legend-label {
  font-size: 11px;
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
}

/* ============ Day view ============ */
.day-grid { display: grid; grid-template-columns: 1fr 320px; gap: 16px; }
@media (max-width: 1100px) { .day-grid { grid-template-columns: 1fr; } }
.day-summary {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.day-summary-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.day-summary-row:last-child { border-bottom: none; padding-bottom: 0; }
.day-summary-label {
  font-size: 11px; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.day-summary-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px; font-weight: 500;
}

/* ============ Anti-leak ============ */
.leak-status-banner {
  border-radius: var(--radius);
  padding: 14px 18px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 14px;
}
.leak-status-banner.ok {
  background: hsla(140, 60%, 90%, 0.6);
  border: 1px solid hsl(140, 50%, 55%);
  color: hsl(140, 70%, 20%);
}
.leak-status-banner.warn {
  background: hsla(0, 70%, 92%, 0.6);
  border: 1px solid hsl(0, 70%, 55%);
  color: hsl(0, 70%, 25%);
}
[data-theme="dark"] .leak-status-banner.ok {
  background: hsla(140, 50%, 18%, 0.6);
  border: 1px solid hsl(140, 50%, 40%);
  color: hsl(140, 60%, 80%);
}
[data-theme="dark"] .leak-status-banner.warn {
  background: hsla(0, 50%, 18%, 0.6);
  border: 1px solid hsl(0, 60%, 50%);
  color: hsl(0, 70%, 80%);
}
.leak-status-icon {
  font-size: 28px;
  font-weight: 700;
  line-height: 1;
  font-family: 'JetBrains Mono', monospace;
}
.leak-status-text { flex: 1; font-size: 13px; line-height: 1.5; }
.leak-status-text strong { font-size: 15px; display: block; margin-bottom: 2px; }

.leak-rule {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-left: 3px solid var(--color-model);
  border-radius: var(--radius);
  padding: 14px 18px;
  margin-bottom: 16px;
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--text);
}
.leak-rule code {
  background: var(--bg-sunken);
  padding: 1px 6px;
  border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}

.status-pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 10px;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.status-safe { background: hsla(140, 60%, 85%, 0.7); color: hsl(140, 70%, 25%); }
.status-deterministic { background: hsla(220, 60%, 85%, 0.7); color: hsl(220, 70%, 30%); }
.status-forecast { background: hsla(180, 50%, 85%, 0.7); color: hsl(180, 70%, 25%); }
.status-leak_suspect { background: hsla(0, 70%, 88%, 0.7); color: hsl(0, 70%, 30%); }
.status-unknown { background: hsla(40, 30%, 85%, 0.7); color: hsl(40, 50%, 30%); }
[data-theme="dark"] .status-safe { background: hsla(140, 50%, 25%, 0.7); color: hsl(140, 70%, 75%); }
[data-theme="dark"] .status-deterministic { background: hsla(220, 50%, 25%, 0.7); color: hsl(220, 70%, 75%); }
[data-theme="dark"] .status-forecast { background: hsla(180, 50%, 25%, 0.7); color: hsl(180, 70%, 75%); }
[data-theme="dark"] .status-leak_suspect { background: hsla(0, 60%, 30%, 0.7); color: hsl(0, 70%, 80%); }
[data-theme="dark"] .status-unknown { background: hsla(40, 30%, 25%, 0.7); color: hsl(40, 60%, 80%); }

/* ============ V5 explainer (transformation de cible) ============ */
.v5-block {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-left: 3px solid var(--color-model);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: 16px;
  font-size: 13px;
  line-height: 1.6;
}
.v5-block h3 {
  font-family: 'Fraunces', serif;
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--text);
}
.v5-block p { margin-bottom: 8px; color: var(--text); }
.v5-block p:last-child { margin-bottom: 0; }
.v5-block code {
  background: var(--bg-sunken);
  padding: 1px 6px;
  border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}
.v5-block ul {
  margin: 8px 0 4px 22px;
  color: var(--text);
}
.v5-block ul li { margin-bottom: 4px; }
.v5-block strong { color: var(--text); }

.v5-pipeline {
  background: var(--bg-sunken);
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius);
  padding: 18px;
  margin: 12px 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
}
.v5-pipeline-row {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  justify-content: center;
}
.v5-pipeline-box {
  background: var(--bg-elev);
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  padding: 6px 12px;
  min-width: 90px;
  text-align: center;
}
.v5-pipeline-box.target {
  border-color: var(--color-model);
  background: var(--color-model);
  color: white;
  font-weight: 600;
}
.v5-pipeline-box.input { border-color: var(--color-baseline); }
.v5-pipeline-box.output { border-color: var(--color-corrected); }
.v5-pipeline-arrow {
  color: var(--text-muted);
  font-size: 16px;
}
.v5-pipeline-step {
  font-size: 10px;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 2px;
}
[data-theme="dark"] .v5-pipeline { background: var(--bg-sunken); }

/* Cache les blocs V5 par defaut, le JS les active si isV5 */
.v5-only { display: none; }

.footer {
  padding: 24px;
  text-align: center;
  font-size: 11px;
  color: var(--text-faint);
  border-top: 1px solid var(--border);
  margin-top: 32px;
  font-family: 'JetBrains Mono', monospace;
}

.toggle-list {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.toggle-item {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 6px 10px;
  background: var(--bg);
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  user-select: none;
  transition: background 0.12s, border-color 0.12s;
}
.toggle-item:hover {
  background: var(--bg-sunken);
  border-color: var(--color-accent);
}
.toggle-item input[type="checkbox"] {
  appearance: none;
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border: 1.5px solid var(--border-strong);
  border-radius: 3px;
  cursor: pointer;
  position: relative;
  flex-shrink: 0;
  margin: 0;
}
.toggle-item input[type="checkbox"]:checked {
  background: var(--color-accent);
  border-color: var(--color-accent);
}
.toggle-item input[type="checkbox"]:checked::after {
  content: "\2713";
  color: white;
  font-size: 11px;
  font-weight: 700;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  line-height: 1;
}
.toggle-color {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
  border: 1px solid rgba(0,0,0,0.1);
}
.toggle-item.dimmed { opacity: 0.55; }

.no-model-banner {
  background: hsla(40, 80%, 92%, 0.6);
  border: 1px solid hsl(40, 70%, 60%);
  color: hsl(40, 70%, 25%);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin: 24px;
  text-align: center;
  font-size: 14px;
}
[data-theme="dark"] .no-model-banner {
  background: hsla(40, 50%, 18%, 0.6);
  border: 1px solid hsl(40, 60%, 45%);
  color: hsl(40, 70%, 80%);
}
</style>
</head>
<body data-theme="light" data-model="__DEFAULT_MODEL__">

<div class="header">
  <div class="brand">
    <div class="brand-title">OIKEN ML, dashboard</div>
    <div class="brand-meta" id="brand-meta">-</div>
  </div>
  <div class="header-stats">
    <div class="header-stat"><strong id="hs-days">-</strong> jours</div>
    <div class="header-stat"><strong id="hs-pts">-</strong> points</div>
    <div class="header-stat"><strong id="hs-features">-</strong> features</div>
  </div>
  <div class="model-switcher" id="model-switcher"></div>
  <button class="theme-toggle" id="theme-toggle">Theme sombre</button>
</div>

<div class="tabs" id="tabs">
  <button class="tab active" data-panel="synthese"><span class="tab-num">01</span>Synthese</button>
  <button class="tab" data-panel="mois"><span class="tab-num">02</span>Mois</button>
  <button class="tab" data-panel="jour"><span class="tab-num">03</span>Jour</button>
  <button class="tab" data-panel="diagnostics"><span class="tab-num">04</span>Diagnostics</button>
  <button class="tab" data-panel="drift"><span class="tab-num">05</span><span id="tab-drift-label">Drift</span></button>
  <button class="tab" data-panel="features"><span class="tab-num">06</span>Features</button>
  <button class="tab" data-panel="leak"><span class="tab-num">07</span>Anti-leak</button>
</div>

<div class="content">

<!-- PANEL SYNTHESE -->
<div class="panel active" id="panel-synthese">
  <h1 class="panel-title">Synthese</h1>
  <p class="panel-subtitle">Performance globale du modele <span class="model-tag" id="syn-model-tag">-</span> sur 5 folds (expanding window).</p>
  <div class="cards cols-4" id="kpi-cards"></div>
  <div style="height:16px;"></div>
  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">MAE par fold (load reconstruit)</div><div id="plot-mae-folds" style="height:340px;"></div></div>
    <div class="plot"><div class="plot-title">Gain MAE vs baseline OIKEN par fold</div><div id="plot-gain-folds" style="height:340px;"></div></div>
  </div>
  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">Distribution des erreurs (modele vs baseline)</div><div id="plot-error-dist" style="height:380px;"></div></div>
    <div class="plot"><div class="plot-title">Tableau detaille par fold</div><div class="tbl-wrap"><table class="tbl" id="tbl-folds"></table></div></div>
  </div>
</div>

<!-- PANEL MOIS -->
<div class="panel" id="panel-mois">
  <h1 class="panel-title">Vue mensuelle</h1>
  <p class="panel-subtitle">Carte calendaire MAE par jour pour <span class="model-tag" id="mois-model-tag">-</span>. Vert = bon, rouge = mauvais. Cliquer un jour pour basculer en vue jour.</p>
  <div class="controls">
    <span class="controls-label">Metrique affichee</span>
    <div class="control-group">
      <button class="btn active" data-metric="mae_model">Modele</button>
      <button class="btn" data-metric="mae_baseline">Baseline</button>
      <button class="btn" data-metric="gain">Gain (%)</button>
    </div>
  </div>
  <div class="cal-legend">
    <span class="cal-legend-label" id="cal-legend-min">min</span>
    <div class="cal-legend-bar"></div>
    <span class="cal-legend-label" id="cal-legend-max">max</span>
    <span class="cal-legend-label" style="margin-left:auto;">vert = bonne prediction, rouge = mauvaise</span>
  </div>
  <div class="calendar-wrap" id="calendar-wrap"></div>
  <div style="height:16px;"></div>
  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">Profil moyen 24h par mois</div><div id="plot-monthly-profile" style="height:380px;"></div></div>
    <div class="plot"><div class="plot-title">MAE moyenne par mois (modele vs baseline)</div><div id="plot-monthly-mae" style="height:380px;"></div></div>
  </div>
</div>

<!-- PANEL JOUR -->
<div class="panel" id="panel-jour">
  <h1 class="panel-title">Vue journaliere</h1>
  <p class="panel-subtitle">96 pas de 15 min. Comparaison reel / baseline OIKEN / <span class="model-tag" id="jour-model-tag">-</span> / modele corrige. Axe de droite : irradiance solaire (W/m2).</p>
  <div class="controls">
    <span class="controls-label">Date</span>
    <select class="control" id="day-select"></select>
    <button class="btn" id="day-prev">&lt; prec</button>
    <button class="btn" id="day-next">suiv &gt;</button>
    <span class="controls-label" style="margin-left:24px;">Traces</span>
    <div class="toggle-list">
      <label class="toggle-item">
        <input type="checkbox" data-trace="real" checked>
        <span class="toggle-color" style="background: var(--color-real);"></span>Reel
      </label>
      <label class="toggle-item">
        <input type="checkbox" data-trace="baseline" checked>
        <span class="toggle-color" style="background: var(--color-baseline);"></span>Baseline
      </label>
      <label class="toggle-item">
        <input type="checkbox" data-trace="model" checked>
        <span class="toggle-color" id="toggle-color-model" style="background: var(--color-model);"></span>Modele
      </label>
      <label class="toggle-item dimmed">
        <input type="checkbox" data-trace="corrected">
        <span class="toggle-color" style="background: var(--color-corrected);"></span>Corrige
      </label>
      <label class="toggle-item">
        <input type="checkbox" data-trace="irradiance" checked>
        <span class="toggle-color" style="background: var(--color-irradiance);"></span>Irradiance
      </label>
    </div>
  </div>
  <div class="day-grid">
    <div>
      <div class="plot"><div class="plot-title" id="day-plot-title">Charge predite vs reelle</div><div id="plot-day-main" style="height:460px;"></div></div>
      <div class="plot"><div class="plot-title">Erreurs absolues (residus)</div><div id="plot-day-errors" style="height:220px;"></div></div>
    </div>
    <div class="day-summary" id="day-summary"></div>
  </div>
</div>

<!-- PANEL DIAGNOSTICS -->
<div class="panel" id="panel-diagnostics">
  <h1 class="panel-title">Diagnostics</h1>
  <p class="panel-subtitle">Decomposition de l'erreur par heure et jour de semaine pour <span class="model-tag" id="diag-model-tag">-</span>.</p>
  <div class="controls">
    <span class="controls-label">Fold</span>
    <select class="control" id="diag-fold-select">
      <option value="all">Tous (agrege)</option>
      <option value="1">Fold 1</option>
      <option value="2">Fold 2</option>
      <option value="3">Fold 3</option>
      <option value="4">Fold 4</option>
      <option value="5">Fold 5</option>
    </select>
  </div>
  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">MAE par heure de la journee</div><div id="plot-diag-hour" style="height:360px;"></div></div>
    <div class="plot"><div class="plot-title">MAE par jour de semaine</div><div id="plot-diag-dow" style="height:360px;"></div></div>
  </div>
  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">Scatter : prediction vs verite (modele)</div><div id="plot-diag-scatter" style="height:420px;"></div></div>
    <div class="plot"><div class="plot-title">Distribution des residus signes (load_pred - load_true)</div><div id="plot-diag-residual-hist" style="height:420px;"></div></div>
  </div>
</div>

<!-- PANEL DRIFT / RESIDU PV (adaptatif V5 vs legacy) -->
<div class="panel" id="panel-drift">
  <h1 class="panel-title" id="drift-panel-title">Correction additive (drift PV)</h1>
  <p class="panel-subtitle" id="drift-panel-subtitle">Pente, intercept et impact effectif de la correction par fold pour <span class="model-tag" id="drift-model-tag">-</span>.</p>

  <!-- Bloc explicatif V5 : pourquoi cette transformation existe -->
  <div class="v5-only v5-block" id="v5-explainer-problem">
    <h3>Le problème : XGBoost sature aux pas de forte production PV</h3>
    <p>La consommation nette d'OIKEN devient parfois très basse (voire négative) en plein midi d'été, quand la production PV locale dépasse la demande. Les arbres de décision <strong>n'extrapolent pas</strong> : ils plafonnent aux bornes vues à l'entraînement et produisent une prédiction systématiquement trop élevée dans cette zone.</p>
    <p>Conséquence : l'erreur du modèle augmente fortement aux heures ensoleillées, et grossit avec le temps puisque la puissance PV installée croît d'année en année.</p>
  </div>

  <div class="v5-only v5-block" id="v5-explainer-solution">
    <h3>La solution V5 : on retire l'effet PV de la cible <em>avant</em> d'entraîner</h3>
    <p>Plutôt que d'apprendre la consommation brute puis de la corriger après coup (V4), on apprend une cible déjà débarrassée de la composante linéaire du PV. Le modèle se concentre alors sur la dynamique résiduelle (température, calendaire, lags), que les arbres traitent bien.</p>
    <div class="v5-pipeline" id="v5-pipeline-diagram">
      <div class="v5-pipeline-row">
        <div>
          <div class="v5-pipeline-box input">load (réel)</div>
          <div class="v5-pipeline-step">cible originale</div>
        </div>
        <div class="v5-pipeline-arrow">−</div>
        <div>
          <div class="v5-pipeline-box">γ × pv_predicted_kwh</div>
          <div class="v5-pipeline-step">effet PV estimé par OLS</div>
        </div>
        <div class="v5-pipeline-arrow">=</div>
        <div>
          <div class="v5-pipeline-box target">target</div>
          <div class="v5-pipeline-step">cible apprise par XGBoost</div>
        </div>
      </div>
      <div style="color: var(--text-faint); font-size:10px;">▼ entraînement, puis à l'inférence ▼</div>
      <div class="v5-pipeline-row">
        <div>
          <div class="v5-pipeline-box target">target_pred</div>
          <div class="v5-pipeline-step">sortie XGBoost</div>
        </div>
        <div class="v5-pipeline-arrow">+</div>
        <div>
          <div class="v5-pipeline-box">γ × pv_predicted_kwh</div>
          <div class="v5-pipeline-step">recompose l'effet PV</div>
        </div>
        <div class="v5-pipeline-arrow">=</div>
        <div>
          <div class="v5-pipeline-box output">load_pred</div>
          <div class="v5-pipeline-step">prédiction finale</div>
        </div>
      </div>
    </div>
    <p><strong>Estimation de γ</strong> : régression linéaire (OLS) de <code>load</code> sur <code>pv_predicted_kwh</code>, restreinte aux pas <em>diurnaux</em> (où la production PV est significative, au-dessus d'un seuil) et trimée pour ignorer les valeurs aberrantes. γ est ré-estimé à chaque fold sur les données de train uniquement.</p>
  </div>

  <div class="cards cols-3" id="drift-kpi-cards">
    <div class="card">
      <div class="card-title" id="drift-card-1-title">Forme</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size:12px;" id="drift-form">-</div>
    </div>
    <div class="card">
      <div class="card-title" id="drift-card-2-title">Tail days calibration</div>
      <div class="kpi-value" id="drift-tail">-</div>
      <div class="kpi-sub" id="drift-card-2-sub">jours en fin de train</div>
    </div>
    <div class="card">
      <div class="card-title" id="drift-card-3-title">Amplitude clip</div>
      <div class="kpi-value" id="drift-clip">-</div>
      <div class="kpi-sub" id="drift-card-3-sub">borne max d'extrapolation</div>
    </div>
  </div>

  <!-- Guide de lecture des chiffres (V5 seulement) -->
  <div class="v5-only v5-block" id="v5-explainer-readme">
    <h3>Comment lire le tableau et les graphiques ci-dessous</h3>
    <ul>
      <li><strong>γ (résidu)</strong> : pente estimée. Doit être <em>négative</em> car plus le PV produit, moins OIKEN doit acheter sur le réseau. L'ordre de grandeur attendu est ~10⁻⁴ (load standardisé / kWh PV).</li>
      <li><strong>R² OLS</strong> : qualité du fit linéaire load ~ γ × pv_proxy sur les pas diurnaux. Plus c'est haut, plus l'effet PV est bien capturé linéairement. Au-dessus de 0.05 le fit est exploitable, au-dessus de 0.10 il est solide.</li>
      <li><strong>n diurnal</strong> : nombre de pas de 15 min utilisés pour estimer γ (ceux où le PV prévu dépasse le seuil).</li>
      <li><strong>Seuil PV</strong> : borne en kWh au-dessus de laquelle un pas est considéré diurnal pour l'estimation OLS.</li>
      <li><strong>MAE V5 vs MAE baseline</strong> : si MAE V5 &lt; MAE baseline, la transformation V5 améliore réellement la prédiction sur ce fold.</li>
      <li><strong>Fit OLS = OK</strong> : R² supérieur à 0.05 (seuil arbitraire). Un OK indique seulement que la régression est exploitable, pas que la prédiction finale est bonne.</li>
    </ul>
  </div>

  <div class="plot">
    <div class="plot-title" id="drift-table-title">Tableau drift par fold</div>
    <div class="tbl-wrap"><table class="tbl" id="tbl-drift"></table></div>
  </div>
  <div class="cards cols-2">
    <div class="plot"><div class="plot-title" id="drift-plot1-title">Pente estimee par fold</div><div id="plot-drift-slope" style="height:340px;"></div></div>
    <div class="plot"><div class="plot-title" id="drift-plot2-title">R2 du fit lineaire par fold</div><div id="plot-drift-r2" style="height:340px;"></div></div>
  </div>
</div>

<!-- PANEL FEATURES -->
<div class="panel" id="panel-features">
  <h1 class="panel-title">Importance des features</h1>
  <p class="panel-subtitle">Top 30 du modele final <span class="model-tag" id="feat-model-tag">-</span> (importance gain native). Survoler une barre pour voir le nom technique.</p>
  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">Top 30 features (gain)</div><div id="plot-fi-bar" style="height:680px;"></div></div>
    <div class="plot"><div class="plot-title">Importance agregee par categorie</div><div id="plot-fi-category" style="height:340px;"></div>
      <div class="tbl-wrap" style="margin-top:12px;"><table class="tbl" id="tbl-fi-category"></table></div>
    </div>
  </div>
</div>

<!-- PANEL ANTI-LEAK (frise temporelle supprimee) -->
<div class="panel" id="panel-leak">
  <h1 class="panel-title">Audit anti-leak</h1>
  <p class="panel-subtitle">Verification que le modele <span class="model-tag" id="leak-model-tag">-</span> n'utilise aucune donnee future qui ne serait pas disponible au moment du run de prevision.</p>
  <div id="leak-status-banner-wrap"></div>
  <div class="leak-rule" id="leak-rule"></div>
  <div class="cards cols-2">
    <div class="plot">
      <div class="plot-title">Repartition des statuts dans le top 30</div>
      <div id="plot-leak-status" style="height:340px;"></div>
    </div>
    <div class="plot">
      <div class="plot-title">Validation du pipeline Python</div>
      <div id="leak-validation-output" style="font-size:12px; line-height:1.6;"></div>
    </div>
  </div>
  <div class="plot">
    <div class="plot-title">Audit feature par feature (top 30)</div>
    <div class="tbl-wrap"><table class="tbl" id="tbl-leak-features"></table></div>
  </div>
</div>

</div><!-- /content -->

<div class="footer" id="footer">
  Generated <span id="footer-time"></span> | OIKEN ML | <span id="footer-model">-</span>
</div>

<script id="data-payload" type="application/json">__DATA_JSON__</script>

__JS_BLOCK__

</body>
</html>
"""


# ============================================================
# Bloc JavaScript
# ============================================================

JS_BLOCK = r"""<script>
"use strict";

const DATA = JSON.parse(document.getElementById("data-payload").textContent);
const PLOTLY_FONT = { family: "'IBM Plex Sans', sans-serif", size: 11 };
const PLOTLY_CONFIG = {
  displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"]
};

// ============ Modele actif ============
let currentModel = DATA.default_model;

function M() {
  return DATA.models[currentModel];
}
function MMeta() {
  return DATA.model_meta[currentModel];
}
function getModelColor() {
  const meta = MMeta();
  if (!meta) return "#888";
  const theme = document.body.getAttribute("data-theme");
  return theme === "dark" ? meta.color_dark : meta.color_light;
}
function applyModelColor() {
  document.documentElement.style.setProperty("--color-model",
    getModelColor());
}

// ============ Theme ============
function getCssVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function plotlyLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: Object.assign({}, PLOTLY_FONT, { color: getCssVar("--text") }),
    margin: { l: 50, r: 18, t: 28, b: 38 },
    xaxis: { gridcolor: getCssVar("--border"),
             zerolinecolor: getCssVar("--border-strong"),
             tickfont: { size: 10 } },
    yaxis: { gridcolor: getCssVar("--border"),
             zerolinecolor: getCssVar("--border-strong"),
             tickfont: { size: 10 } },
    legend: { orientation: "h", y: -0.15, x: 0, xanchor: "left",
              font: { size: 11 } },
    hovermode: "x unified",
  }, extra);
}

const themeToggle = document.getElementById("theme-toggle");
themeToggle.addEventListener("click", () => {
  const cur = document.body.getAttribute("data-theme");
  const next = cur === "light" ? "dark" : "light";
  document.body.setAttribute("data-theme", next);
  themeToggle.textContent = next === "light" ? "Theme sombre" : "Theme clair";
  applyModelColor();
  setTimeout(renderAll, 50);
});

// ============ Switcher de modele ============
function initModelSwitcher() {
  const sw = document.getElementById("model-switcher");
  const available = DATA.available_models;
  if (!available || available.length === 0) {
    sw.classList.add("hidden");
    return;
  }
  if (available.length === 1) {
    sw.classList.add("hidden");
    return;
  }
  sw.innerHTML = available.map(key => {
    const meta = DATA.model_meta[key];
    const active = key === currentModel ? "active" : "";
    return `<button class="model-btn ${active}" data-model="${key}">
      <span class="model-btn-dot"></span>${meta.display_name}
    </button>`;
  }).join("");
  sw.querySelectorAll(".model-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const newModel = btn.dataset.model;
      if (newModel === currentModel) return;
      currentModel = newModel;
      document.body.setAttribute("data-model", currentModel);
      sw.querySelectorAll(".model-btn").forEach(b =>
        b.classList.toggle("active", b.dataset.model === currentModel));
      applyModelColor();
      updateHeaderAndTags();
      setTimeout(renderAll, 30);
    });
  });
}

// ============ Tabs ============
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t =>
      t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p =>
      p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + tab.dataset.panel)
      .classList.add("active");
    setTimeout(renderAll, 30);
  });
});

// ============ Header + tags ============
function updateHeaderAndTags() {
  const m = M();
  const meta = MMeta();
  if (!m || !meta) return;
  document.getElementById("hs-days").textContent =
    m.meta.n_days.toLocaleString("fr-CH");
  document.getElementById("hs-pts").textContent =
    m.meta.n_predictions.toLocaleString("fr-CH");
  document.getElementById("hs-features").textContent = m.meta.n_features;
  document.getElementById("brand-meta").textContent =
    `${meta.display_name} ${m.meta.version} | ${m.meta.ts_min} -> ${m.meta.ts_max}`;
  document.getElementById("footer-time").textContent =
    DATA.global_meta.generated_at;
  document.getElementById("footer-model").textContent = meta.display_name;
  ["syn", "mois", "jour", "diag", "drift", "feat", "leak"].forEach(p => {
    const el = document.getElementById(p + "-model-tag");
    if (el) el.textContent = meta.display_name;
  });
}

// ============ Helpers numeriques ============
function fmt(n, dec = 4) {
  if (n === null || n === undefined || isNaN(n)) return "-";
  return Number(n).toFixed(dec);
}
function meanArr(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// ============ Palette HSL pour le calendrier ============
function maeColor(value, min, max, theme) {
  if (value === null || value === undefined || isNaN(value))
    return "transparent";
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const hue = 120 * (1 - t);
  const sat = 70;
  const light = theme === "dark" ? 38 : 50;
  const alpha = theme === "dark" ? 0.78 : 0.65;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}
function gainColor(value, theme) {
  if (value === null || value === undefined || isNaN(value))
    return "transparent";
  const t = Math.max(0, Math.min(1, (value + 50) / 100));
  const hue = 120 * t;
  const sat = 70;
  const light = theme === "dark" ? 38 : 50;
  const alpha = theme === "dark" ? 0.78 : 0.65;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

// ============ SYNTHESE ============
function renderSynthese() {
  const m = M();
  if (!m) return;
  const folds = m.folds;
  const allMaeModel = folds.map(f => f.metrics_no_correction.mae_load);
  const allMaeBase = folds.map(f => f.metrics_no_correction.mae_baseline);
  const allRmseModel = folds.map(f => f.metrics_no_correction.rmse_load);
  const allRmseBase = folds.map(f => f.metrics_no_correction.rmse_baseline);
  const allMape = folds.map(f => f.metrics_no_correction.mape_load_pct);
  const meanGain = meanArr(folds.map(f =>
    f.metrics_no_correction.gain_mae_vs_baseline));
  const improve = (1 - meanGain) * 100;
  const modelName = MMeta().display_name;

  const cards = [
    { title: `MAE moyenne (${modelName})`, value: fmt(meanArr(allMaeModel)),
      sub: `vs baseline ${fmt(meanArr(allMaeBase))}` },
    { title: `RMSE moyenne (${modelName})`, value: fmt(meanArr(allRmseModel)),
      sub: `vs baseline ${fmt(meanArr(allRmseBase))}` },
    { title: "MAPE moyenne", value: fmt(meanArr(allMape), 1) + "%",
      sub: "tous folds confondus" },
    { title: "Amelioration vs baseline", value: improve.toFixed(1) + "%",
      sub: "MAE relative gagnee", improve },
  ];
  document.getElementById("kpi-cards").innerHTML = cards.map(c => {
    const cls = c.improve !== undefined && c.improve > 0 ? "delta-neg" : "";
    return `<div class="card">
      <div class="card-title">${c.title}</div>
      <div class="kpi-value">${c.value}</div>
      <div class="kpi-sub ${cls}">${c.sub}</div>
    </div>`;
  }).join("");

  const colorModel = getCssVar("--color-model");
  Plotly.newPlot("plot-mae-folds", [
    { x: folds.map(f => "Fold " + f.fold), y: allMaeBase, type: "bar",
      name: "Baseline OIKEN",
      marker: { color: getCssVar("--color-baseline") } },
    { x: folds.map(f => "Fold " + f.fold), y: allMaeModel, type: "bar",
      name: modelName, marker: { color: colorModel } },
  ], plotlyLayout({
    barmode: "group",
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "MAE (load standardise)", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);

  const gains = folds.map(f =>
    (1 - f.metrics_no_correction.gain_mae_vs_baseline) * 100);
  Plotly.newPlot("plot-gain-folds", [
    { x: folds.map(f => "Fold " + f.fold), y: gains, type: "bar",
      marker: { color: gains.map(g => g > 20 ? colorModel
                                       : g > 0 ? getCssVar("--color-corrected")
                                       : getCssVar("--color-error")) },
      text: gains.map(g => g.toFixed(1) + "%"), textposition: "outside" },
  ], plotlyLayout({
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "Gain MAE relatif (%)", font: { size: 11 } },
      zeroline: true }),
    showlegend: false,
  }), PLOTLY_CONFIG);

  const allErrModel = [], allErrBase = [];
  Object.values(m.by_day).forEach(day => {
    for (let i = 0; i < day.load_true.length; i++) {
      allErrModel.push(day.load_pred_raw[i] - day.load_true[i]);
      allErrBase.push(day.forecast_baseline[i] - day.load_true[i]);
    }
  });
  Plotly.newPlot("plot-error-dist", [
    { x: allErrBase, type: "histogram", name: "Baseline OIKEN",
      marker: { color: getCssVar("--color-baseline") }, opacity: 0.55,
      xbins: { size: 0.025 } },
    { x: allErrModel, type: "histogram", name: modelName,
      marker: { color: colorModel }, opacity: 0.7,
      xbins: { size: 0.025 } },
  ], plotlyLayout({
    barmode: "overlay",
    xaxis: Object.assign(plotlyLayout().xaxis, {
      title: { text: "Erreur signee (pred - true)", font: { size: 11 } },
      range: [-1.5, 1.5] }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "Frequence", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);

  const rows = folds.map(f => {
    const mt = f.metrics_no_correction;
    const gain = (1 - mt.gain_mae_vs_baseline) * 100;
    return `<tr>
      <td>Fold ${f.fold}</td>
      <td class="num">${f.test_start.slice(0,10)}</td>
      <td class="num">${f.test_end.slice(0,10)}</td>
      <td class="num">${fmt(mt.mae_baseline)}</td>
      <td class="num">${fmt(mt.mae_load)}</td>
      <td class="num">${fmt(mt.rmse_load)}</td>
      <td class="num">${fmt(mt.mape_load_pct, 1)}%</td>
      <td class="num" style="color:${gain > 20 ? 'var(--color-model)'
        : gain > 0 ? 'var(--color-corrected)'
        : 'var(--color-error)'}">${gain.toFixed(1)}%</td>
    </tr>`;
  }).join("");
  document.getElementById("tbl-folds").innerHTML = `
    <thead><tr>
      <th>Fold</th><th>Debut</th><th>Fin</th>
      <th>MAE base</th><th>MAE modele</th><th>RMSE</th><th>MAPE</th><th>Gain</th>
    </tr></thead><tbody>${rows}</tbody>`;
}

// ============ MOIS ============
let currentMonthMetric = "mae_model";
function getDailyMap() {
  const m = M();
  const map = new Map();
  const d = m.daily;
  for (let i = 0; i < d.date.length; i++) {
    map.set(d.date[i], {
      mae_model: d.mae_model[i],
      mae_baseline: d.mae_baseline[i],
      mae_corr: d.mae_corr[i],
      gain: ((d.mae_baseline[i] - d.mae_model[i]) / d.mae_baseline[i]) * 100,
      fold: d.fold[i],
      load_mean: d.load_mean[i],
    });
  }
  return map;
}
function renderCalendar() {
  const dailyMap = getDailyMap();
  const dates = Array.from(dailyMap.keys()).sort();
  if (!dates.length) return;
  const allVals = Array.from(dailyMap.values())
    .map(v => currentMonthMetric === "gain" ? v.gain : v[currentMonthMetric])
    .filter(v => v !== null && v !== undefined && !isNaN(v));
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const theme = document.body.getAttribute("data-theme");

  if (currentMonthMetric === "gain") {
    document.getElementById("cal-legend-min").textContent = "perte (-50%+)";
    document.getElementById("cal-legend-max").textContent = "gain (+50%+)";
  } else {
    document.getElementById("cal-legend-min").textContent =
      `min ${minVal.toFixed(3)}`;
    document.getElementById("cal-legend-max").textContent =
      `max ${maxVal.toFixed(3)}`;
  }

  const byMonth = new Map();
  dates.forEach(d => {
    const k = d.slice(0, 7);
    if (!byMonth.has(k)) byMonth.set(k, []);
    byMonth.get(k).push(d);
  });

  const monthsFr = ["", "Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin",
                    "Juillet", "Aout", "Septembre", "Octobre", "Novembre",
                    "Decembre"];
  const wrap = document.getElementById("calendar-wrap");
  wrap.style.gridTemplateColumns = `repeat(auto-fit, minmax(280px, 1fr))`;
  let html = "";
  byMonth.forEach((days, ymKey) => {
    const [year, month] = ymKey.split("-").map(Number);
    const monthName = `${monthsFr[month]} ${year}`;
    const monthMaes = days.map(d => dailyMap.get(d)[
      currentMonthMetric === "gain" ? "gain" : currentMonthMetric])
      .filter(v => !isNaN(v));
    const monthMean = meanArr(monthMaes);
    const first = new Date(year, month - 1, 1);
    const firstDow = (first.getDay() + 6) % 7;
    const daysInMonth = new Date(year, month, 0).getDate();
    let cells = "";
    for (let i = 0; i < firstDow; i++) {
      cells += `<div class="cal-day empty"></div>`;
    }
    for (let day = 1; day <= daysInMonth; day++) {
      const dStr = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
      const data = dailyMap.get(dStr);
      if (!data) {
        cells += `<div class="cal-day empty"><span class="cal-day-num" style="opacity:0.3">${day}</span></div>`;
      } else {
        const val = currentMonthMetric === "gain" ? data.gain
                    : data[currentMonthMetric];
        const bg = currentMonthMetric === "gain"
          ? gainColor(val, theme)
          : maeColor(val, minVal, maxVal, theme);
        const display = currentMonthMetric === "gain"
          ? (val > 0 ? "+" : "") + val.toFixed(0) + "%"
          : val.toFixed(3);
        const tooltip = `${dStr}\nMAE modele: ${data.mae_model.toFixed(4)}\nMAE baseline: ${data.mae_baseline.toFixed(4)}\nGain: ${data.gain.toFixed(1)}%\nFold ${data.fold}`;
        cells += `<div class="cal-day" style="background:${bg};" data-date="${dStr}" title="${tooltip}">
          <span class="cal-day-num">${day}</span>
          <span class="cal-day-mae">${display}</span>
        </div>`;
      }
    }
    html += `<div class="calendar-month">
      <div class="cal-header">
        <span class="cal-month-name">${monthName}</span>
        <span class="cal-month-mae">moy ${currentMonthMetric === "gain" ? monthMean.toFixed(1) + "%" : monthMean.toFixed(4)}</span>
      </div>
      <div class="cal-grid">
        <div class="cal-dow">L</div><div class="cal-dow">M</div><div class="cal-dow">M</div>
        <div class="cal-dow">J</div><div class="cal-dow">V</div><div class="cal-dow">S</div><div class="cal-dow">D</div>
        ${cells}
      </div>
    </div>`;
  });
  wrap.innerHTML = html;

  wrap.querySelectorAll(".cal-day[data-date]").forEach(el => {
    el.addEventListener("click", () => {
      const d = el.dataset.date;
      document.querySelector('.tab[data-panel="jour"]').click();
      setTimeout(() => {
        const sel = document.getElementById("day-select");
        sel.value = d;
        sel.dispatchEvent(new Event("change"));
      }, 100);
    });
  });
}

function renderMonthlyPlots() {
  const m = M();
  const monthly = m.monthly;
  const byMonth = {};
  Object.entries(m.by_day).forEach(([date, day]) => {
    const ym = date.slice(0, 7);
    if (!byMonth[ym]) byMonth[ym] = {
      hours: Array(96).fill(null).map(() => []) };
    for (let i = 0; i < day.load_true.length && i < 96; i++) {
      byMonth[ym].hours[i].push(day.load_true[i]);
    }
  });
  const traces = [];
  const months = Object.keys(byMonth).sort();
  months.forEach((ym) => {
    const mean = byMonth[ym].hours.map(arr => meanArr(arr));
    const x = Array.from({ length: 96 }, (_, i) =>
      `${String(Math.floor(i/4)).padStart(2,'0')}:${String((i%4)*15).padStart(2,'0')}`);
    traces.push({
      x, y: mean, mode: "lines", name: ym, type: "scatter",
      line: { width: 1.8 },
    });
  });
  Plotly.newPlot("plot-monthly-profile", traces, plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, {
      title: { text: "Heure locale", font: { size: 11 } }, dtick: 16 }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "Charge moyenne", font: { size: 11 } } }),
    legend: { orientation: "h", y: -0.2, font: { size: 10 } },
  }), PLOTLY_CONFIG);

  Plotly.newPlot("plot-monthly-mae", [
    { x: monthly.month, y: monthly.mae_baseline, type: "bar",
      name: "Baseline", marker: { color: getCssVar("--color-baseline") } },
    { x: monthly.month, y: monthly.mae_model, type: "bar",
      name: MMeta().display_name,
      marker: { color: getCssVar("--color-model") } },
  ], plotlyLayout({
    barmode: "group",
    xaxis: Object.assign(plotlyLayout().xaxis, {
      title: { text: "Mois", font: { size: 11 } } }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "MAE", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);
}

document.querySelectorAll('#panel-mois .btn[data-metric]').forEach(b => {
  b.addEventListener("click", () => {
    document.querySelectorAll('#panel-mois .btn[data-metric]').forEach(x =>
      x.classList.remove("active"));
    b.classList.add("active");
    currentMonthMetric = b.dataset.metric;
    renderCalendar();
  });
});

// ============ JOUR ============
let traceVisibility = { real: true, baseline: true, model: true,
                        corrected: false, irradiance: true };

function populateDaySelect() {
  const sel = document.getElementById("day-select");
  const m = M();
  if (!m) return;
  const dates = Object.keys(m.by_day).sort();
  const previousValue = sel.value;
  sel.innerHTML = dates.map(d =>
    `<option value="${d}">${d}</option>`).join("");
  if (previousValue && dates.includes(previousValue)) {
    sel.value = previousValue;
  }
}

function renderDay(date) {
  const m = M();
  if (!m) return;
  const day = m.by_day[date];
  if (!day) return;

  const x = day.hhmm;
  const traces = [];
  const hasIrradiance = day.irradiance && day.irradiance.length > 0
                        && day.irradiance.some(v => v !== null && v > 0);
  const colorModel = getCssVar("--color-model");

  if (traceVisibility.irradiance && hasIrradiance) {
    traces.push({
      x, y: day.irradiance,
      yaxis: "y2",
      mode: "lines", type: "scatter",
      name: m.meta.irradiance_label || "Irradiance (W/m2)",
      line: { color: getCssVar("--color-irradiance"), width: 1.4 },
      fill: "tozeroy",
      fillcolor: "rgba(217, 119, 6, 0.10)",
      hovertemplate: "%{y:.0f} W/m2<extra></extra>",
    });
  }
  if (traceVisibility.real) {
    traces.push({ x, y: day.load_true, mode: "lines", name: "Reel",
      line: { color: getCssVar("--color-real"), width: 2.2 },
      type: "scatter" });
  }
  if (traceVisibility.baseline) {
    traces.push({ x, y: day.forecast_baseline, mode: "lines",
      name: "Baseline OIKEN",
      line: { color: getCssVar("--color-baseline"), width: 1.6, dash: "dot" },
      type: "scatter" });
  }
  if (traceVisibility.model) {
    traces.push({ x, y: day.load_pred_raw, mode: "lines",
      name: MMeta().display_name,
      line: { color: colorModel, width: 1.8 }, type: "scatter" });
  }
  if (traceVisibility.corrected) {
    traces.push({ x, y: day.load_pred_corrected, mode: "lines",
      name: "Modele + drift",
      line: { color: getCssVar("--color-corrected"), width: 1.6,
              dash: "dash" }, type: "scatter" });
  }

  const layout = plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, { dtick: 8, tickangle: 0 }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "Charge standardisee", font: { size: 11 } } }),
    margin: { l: 55, r: 55, t: 12, b: 38 },
  });
  if (hasIrradiance && traceVisibility.irradiance) {
    layout.yaxis2 = {
      title: { text: "Irradiance (W/m2)",
               font: { size: 11, color: getCssVar("--color-irradiance") } },
      overlaying: "y",
      side: "right",
      showgrid: false,
      tickfont: { size: 10, color: getCssVar("--color-irradiance") },
      rangemode: "tozero",
    };
  }
  Plotly.newPlot("plot-day-main", traces, layout, PLOTLY_CONFIG);

  const errModel = day.load_pred_raw.map((v, i) => v - day.load_true[i]);
  const errBase = day.forecast_baseline.map((v, i) => v - day.load_true[i]);
  Plotly.newPlot("plot-day-errors", [
    { x, y: errBase, mode: "lines", name: "Erreur baseline",
      line: { color: getCssVar("--color-baseline"), width: 1.4 },
      type: "scatter" },
    { x, y: errModel, mode: "lines",
      name: "Erreur " + MMeta().display_name,
      line: { color: colorModel, width: 1.6 }, type: "scatter" },
    { x, y: x.map(() => 0), mode: "lines", showlegend: false,
      line: { color: getCssVar("--text-faint"), width: 0.8 },
      type: "scatter" },
  ], plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, { dtick: 8 }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "Erreur signee", font: { size: 11 } },
      zeroline: true }),
    margin: { l: 50, r: 18, t: 12, b: 38 },
  }), PLOTLY_CONFIG);

  const maeModel = meanArr(errModel.map(Math.abs));
  const maeBase = meanArr(errBase.map(Math.abs));
  const rmseModel = Math.sqrt(meanArr(errModel.map(v => v * v)));
  const gain = ((maeBase - maeModel) / maeBase) * 100;
  const loadMean = meanArr(day.load_true);
  const loadMax = Math.max(...day.load_true);
  const loadMin = Math.min(...day.load_true);
  const dt = new Date(date);
  const dowFr = ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi",
                 "Vendredi", "Samedi"][dt.getDay()];
  const gainCls = gain > 20 ? "delta-neg" : gain < 0 ? "delta-pos" : "";
  let irrSummary = "";
  if (hasIrradiance) {
    const irrMax = Math.max(...day.irradiance.filter(v => v !== null));
    const irrSum = day.irradiance.filter(v => v !== null)
      .reduce((a,b) => a+b, 0);
    const irrEnergy = (irrSum * 0.25 / 1000).toFixed(2);
    irrSummary = `
    <div class="day-summary-row">
      <span class="day-summary-label">Irradiance max</span>
      <span class="day-summary-value">${irrMax.toFixed(0)} W/m2</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">Energie solaire jour</span>
      <span class="day-summary-value">${irrEnergy} kWh/m2</span>
    </div>`;
  }
  const summary = `
    <div class="day-summary-row">
      <div>
        <div class="day-summary-label">Date</div>
        <div class="day-summary-value">${date}</div>
        <div style="font-size:11px; color: var(--text-muted);">${dowFr} | Fold ${day.fold} | ${day.n_points} points</div>
      </div>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">MAE ${MMeta().display_name}</span>
      <span class="day-summary-value">${fmt(maeModel)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">MAE baseline</span>
      <span class="day-summary-value">${fmt(maeBase)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">RMSE modele</span>
      <span class="day-summary-value">${fmt(rmseModel)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">Gain MAE</span>
      <span class="day-summary-value ${gainCls}">${gain.toFixed(1)}%</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">Charge moyenne</span>
      <span class="day-summary-value">${fmt(loadMean, 3)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">Charge max</span>
      <span class="day-summary-value">${fmt(loadMax, 3)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">Charge min</span>
      <span class="day-summary-value">${fmt(loadMin, 3)}</span>
    </div>
    ${irrSummary}
  `;
  document.getElementById("day-summary").innerHTML = summary;
  document.getElementById("day-plot-title").textContent =
    `Charge predite vs reelle | ${date} (${dowFr}, fold ${day.fold})`;
}

document.getElementById("day-select").addEventListener("change", e => {
  renderDay(e.target.value);
});
document.getElementById("day-prev").addEventListener("click", () => {
  const sel = document.getElementById("day-select");
  if (sel.selectedIndex > 0) {
    sel.selectedIndex--;
    sel.dispatchEvent(new Event("change"));
  }
});
document.getElementById("day-next").addEventListener("click", () => {
  const sel = document.getElementById("day-select");
  if (sel.selectedIndex < sel.options.length - 1) {
    sel.selectedIndex++;
    sel.dispatchEvent(new Event("change"));
  }
});
document.querySelectorAll('#panel-jour input[type="checkbox"][data-trace]')
  .forEach(cb => {
    cb.addEventListener("change", () => {
      traceVisibility[cb.dataset.trace] = cb.checked;
      cb.parentElement.classList.toggle("dimmed", !cb.checked);
      renderDay(document.getElementById("day-select").value);
    });
  });

// ============ DIAGNOSTICS ============
let diagFold = "all";
function renderDiagnostics() {
  const m = M();
  if (!m) return;
  const h = m.hourly;
  const d = m.dow;
  const colorModel = getCssVar("--color-model");
  const modelName = MMeta().display_name;

  const filterFold = (data) => {
    if (diagFold === "all") return data;
    const idx = data.fold.map((f, i) => f == diagFold ? i : -1)
      .filter(x => x >= 0);
    const out = {};
    Object.keys(data).forEach(k => out[k] = idx.map(i => data[k][i]));
    return out;
  };
  const hf = filterFold(h);
  const hourMap = {};
  hf.hour.forEach((hr, i) => {
    if (!hourMap[hr]) hourMap[hr] = { mae_model: [], mae_baseline: [] };
    hourMap[hr].mae_model.push(hf.mae_model[i]);
    hourMap[hr].mae_baseline.push(hf.mae_baseline[i]);
  });
  const hours = Object.keys(hourMap).sort((a, b) => +a - +b);
  Plotly.newPlot("plot-diag-hour", [
    { x: hours, y: hours.map(h => meanArr(hourMap[h].mae_baseline)),
      type: "bar", name: "Baseline",
      marker: { color: getCssVar("--color-baseline") }, opacity: 0.85 },
    { x: hours, y: hours.map(h => meanArr(hourMap[h].mae_model)),
      type: "bar", name: modelName,
      marker: { color: colorModel }, opacity: 0.95 },
  ], plotlyLayout({
    barmode: "group",
    xaxis: Object.assign(plotlyLayout().xaxis, {
      title: { text: "Heure", font: { size: 11 } }, dtick: 1 }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "MAE", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);

  const df = filterFold(d);
  const dowMap = {};
  df.dow.forEach((dw, i) => {
    if (!dowMap[dw]) dowMap[dw] = { mae_model: [], mae_baseline: [] };
    dowMap[dw].mae_model.push(df.mae_model[i]);
    dowMap[dw].mae_baseline.push(df.mae_baseline[i]);
  });
  const dows = Object.keys(dowMap).sort((a, b) => +a - +b);
  const dowLabels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"];
  Plotly.newPlot("plot-diag-dow", [
    { x: dows.map(d => dowLabels[(+d) - 1] || d),
      y: dows.map(d => meanArr(dowMap[d].mae_baseline)), type: "bar",
      name: "Baseline",
      marker: { color: getCssVar("--color-baseline") }, opacity: 0.85 },
    { x: dows.map(d => dowLabels[(+d) - 1] || d),
      y: dows.map(d => meanArr(dowMap[d].mae_model)), type: "bar",
      name: modelName,
      marker: { color: colorModel }, opacity: 0.95 },
  ], plotlyLayout({
    barmode: "group",
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "MAE", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);

  const allTrue = [], allPred = [];
  Object.entries(m.by_day).forEach(([date, day]) => {
    if (diagFold !== "all" && day.fold != diagFold) return;
    for (let i = 0; i < day.load_true.length; i++) {
      allTrue.push(day.load_true[i]);
      allPred.push(day.load_pred_raw[i]);
    }
  });
  let sampleTrue = allTrue, samplePred = allPred;
  if (allTrue.length > 6000) {
    const stride = Math.ceil(allTrue.length / 6000);
    sampleTrue = allTrue.filter((_, i) => i % stride === 0);
    samplePred = allPred.filter((_, i) => i % stride === 0);
  }
  const minV = Math.min(...sampleTrue, ...samplePred);
  const maxV = Math.max(...sampleTrue, ...samplePred);
  Plotly.newPlot("plot-diag-scatter", [
    { x: sampleTrue, y: samplePred, mode: "markers", type: "scatter",
      marker: { color: colorModel, size: 3, opacity: 0.4 },
      name: "Predictions" },
    { x: [minV, maxV], y: [minV, maxV], mode: "lines", type: "scatter",
      line: { color: getCssVar("--text-faint"), dash: "dash", width: 1 },
      name: "y = x", showlegend: true },
  ], plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, {
      title: { text: "Reel", font: { size: 11 } } }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "Predit (" + modelName + ")", font: { size: 11 } } }),
    hovermode: "closest",
  }), PLOTLY_CONFIG);

  const allRes = allPred.map((p, i) => p - allTrue[i]);
  Plotly.newPlot("plot-diag-residual-hist", [
    { x: allRes, type: "histogram",
      marker: { color: colorModel },
      opacity: 0.85, xbins: { size: 0.02 }, name: "Residus" },
  ], plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, {
      title: { text: "Residu signe (pred - true)", font: { size: 11 } },
      range: [-1.2, 1.2] }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "Frequence", font: { size: 11 } } }),
    showlegend: false,
  }), PLOTLY_CONFIG);
}
document.getElementById("diag-fold-select").addEventListener("change", e => {
  diagFold = e.target.value;
  renderDiagnostics();
});

// ============ DRIFT / RESIDU PV (adaptatif V3/V4/V5) ============
function renderDrift() {
  const m = M();
  if (!m) return;
  const colorModel = getCssVar("--color-model");

  // Detection de la version du modele
  // V5 : presence de residual_target avec final_model_gamma
  // V4 : pv_correction_v4.proxy_col_used non vide
  // V3 : drift_correction temporel legacy uniquement
  const isV5 = m.residual_target && m.residual_target.final_model_gamma
               !== undefined;
  const isV4 = !isV5 && m.pv_correction_v4
               && m.pv_correction_v4.proxy_col_used;

  // ----- En-tete adaptatif -----
  const titleEl = document.getElementById("drift-panel-title");
  const subtitleEl = document.getElementById("drift-panel-subtitle");
  const tabLabelEl = document.getElementById("tab-drift-label");

  // Affichage conditionnel des blocs explicatifs V5
  const v5Blocks = document.querySelectorAll(".v5-only");
  v5Blocks.forEach(b => { b.style.display = isV5 ? "block" : "none"; });

  if (isV5) {
    tabLabelEl.textContent = "Residu PV";
    titleEl.textContent = "Transformation de cible : residu PV (V5)";
    subtitleEl.innerHTML = `Modele <span class="model-tag" id="drift-model-tag">${MMeta().display_name}</span>.
      Cette section explique comment le modele V5 gere l'effet du PV sans utiliser
      de correction de drift au sens classique.`;
  } else if (isV4) {
    tabLabelEl.textContent = "PV correction";
    titleEl.textContent = "Correction PV conditionnelle (V4)";
    subtitleEl.innerHTML = `Correction additive post-hoc conditionnelle au PV prevu pour
      <span class="model-tag" id="drift-model-tag">${MMeta().display_name}</span>.`;
  } else {
    tabLabelEl.textContent = "Drift";
    titleEl.textContent = "Correction additive (drift PV)";
    subtitleEl.innerHTML = `Pente, intercept et impact effectif de la correction par fold pour
      <span class="model-tag" id="drift-model-tag">${MMeta().display_name}</span>.`;
  }

  // ----- KPI cards adaptatifs -----
  if (isV5) {
    const rt = m.residual_target;
    const ginfo = rt.final_model_gamma_info || {};
    document.getElementById("drift-card-1-title").textContent = "Forme";
    document.getElementById("drift-form").textContent =
      `target = load - \u03B3 \u00B7 ${rt.pv_proxy_col || "pv_predicted_kwh"}`;
    document.getElementById("drift-card-2-title").textContent =
      "\u03B3 final (modele entier)";
    document.getElementById("drift-tail").textContent =
      (rt.final_model_gamma || 0).toExponential(3);
    document.getElementById("drift-card-2-sub").textContent =
      `R\u00B2 OLS = ${(ginfo.r2 || 0).toFixed(3)}, n_diurnal = ${ginfo.n_diurnal || "-"}`;
    document.getElementById("drift-card-3-title").textContent = "Methode";
    document.getElementById("drift-clip").textContent = "OLS";
    document.getElementById("drift-card-3-sub").textContent =
      `quantile diurnal = ${rt.diurnal_quantile || "-"}, trim = ${
        (rt.trim_quantiles || []).join("-")}`;
  } else if (isV4) {
    const pc = m.pv_correction_v4 || {};
    document.getElementById("drift-card-1-title").textContent = "Forme";
    document.getElementById("drift-form").textContent =
      pc.form || "additive conditionnelle";
    document.getElementById("drift-card-2-title").textContent =
      "Tail days calibration";
    document.getElementById("drift-tail").textContent =
      pc.tail_days_for_calibration ?? "-";
    document.getElementById("drift-card-2-sub").textContent =
      `proxy = ${pc.proxy_col_used || "-"}, seuil Q${
        ((pc.threshold_quantile_diurnal || 0) * 100).toFixed(0)
      } diurnal`;
    document.getElementById("drift-card-3-title").textContent =
      "Amplitude clip";
    document.getElementById("drift-clip").textContent =
      "+/- " + (pc.amplitude_clip ?? "-");
    document.getElementById("drift-card-3-sub").textContent =
      `trim ${(pc.trim_quantiles || []).map(q =>
        (q * 100).toFixed(0) + "%").join(" / ")}, ` +
      `min ${pc.min_diurnal_samples || "-"} samples`;
  } else {
    const dc = m.drift_correction || {};
    document.getElementById("drift-card-1-title").textContent = "Forme";
    document.getElementById("drift-form").textContent = dc.form || "-";
    document.getElementById("drift-card-2-title").textContent =
      "Tail days calibration";
    document.getElementById("drift-tail").textContent =
      dc.tail_days_for_calibration || "-";
    document.getElementById("drift-card-2-sub").textContent =
      "jours en fin de train";
    document.getElementById("drift-card-3-title").textContent =
      "Amplitude clip";
    document.getElementById("drift-clip").textContent =
      "+/- " + (dc.amplitude_clip || "-");
    document.getElementById("drift-card-3-sub").textContent =
      "borne max d'extrapolation";
  }

  // ----- Tableau par fold -----
  const tableTitle = document.getElementById("drift-table-title");
  const tblEl = document.getElementById("tbl-drift");

  if (isV5) {
    tableTitle.textContent = "Estimation \u03B3 et performance par fold";
    // Bug fix V5 : champs corrects = f.gamma (pas f.gamma_residual),
    // gi.threshold_pv (pas gi.threshold_pv_diurnal). Voir model_XGBoost.py
    // train_fold (return f["gamma"]) et estimate_gamma (info["threshold_pv"]).
    const rows = m.folds.map(f => {
      const gi = f.gamma_info || {};
      const mNo = f.metrics_no_correction;
      const reconstructionOk = gi.r2 !== undefined && gi.r2 > 0.05;
      return `<tr>
        <td>Fold ${f.fold}</td>
        <td class="num">${(f.gamma || 0).toExponential(3)}</td>
        <td class="num">${fmt(gi.r2 || 0, 3)}</td>
        <td class="num">${gi.n_diurnal || "-"}</td>
        <td class="num">${fmt(gi.threshold_pv || 0, 3)}</td>
        <td class="num">${fmt(mNo.mae_baseline)}</td>
        <td class="num">${fmt(mNo.mae_load)}</td>
        <td class="num" style="color:${reconstructionOk ? 'var(--color-model)' : 'var(--color-error)'}">${reconstructionOk ? "OK" : "FAIBLE"}</td>
      </tr>`;
    }).join("");
    tblEl.innerHTML = `
      <thead><tr>
        <th>Fold</th><th>\u03B3 (residu)</th><th>R\u00B2 OLS</th>
        <th>n diurnal</th><th>seuil pv</th>
        <th>MAE baseline</th><th>MAE V5</th><th>Fit OLS</th>
      </tr></thead><tbody>${rows}</tbody>`;
  } else if (isV4) {
    tableTitle.textContent = "Correction PV par fold (V4)";
    const rows = m.folds.map(f => {
      const di = f.drift_info || {};
      const beta = f.pv_correction_beta;
      const alpha = f.pv_correction_alpha;
      const thr = f.pv_correction_threshold;
      const mNo = f.metrics_no_correction;
      const mCo = f.metrics_with_correction;
      const delta = ((mCo.mae_load - mNo.mae_load) * 1000);
      return `<tr>
        <td>Fold ${f.fold}</td>
        <td class="num">${(beta ?? 0).toExponential(2)}</td>
        <td class="num">${fmt(alpha ?? 0)}</td>
        <td class="num">${fmt(thr ?? 0)}</td>
        <td class="num">${fmt(di.r2 || 0, 3)}</td>
        <td class="num">${di.n_diurnal || "-"}</td>
        <td class="num">${fmt(mNo.mae_load)}</td>
        <td class="num">${fmt(mCo.mae_load)}</td>
        <td class="num" style="color:${delta < 0 ? 'var(--color-model)' : delta > 0 ? 'var(--color-error)' : 'var(--text-muted)'}">${delta > 0 ? "+" : ""}${delta.toFixed(2)} e-3</td>
      </tr>`;
    }).join("");
    tblEl.innerHTML = `
      <thead><tr>
        <th>Fold</th><th>\u03B2 (pente)</th><th>\u03B1 (intercept)</th>
        <th>seuil PV</th><th>R\u00B2 OLS</th><th>n diurnal</th>
        <th>MAE sans corr</th><th>MAE avec corr</th><th>Delta MAE</th>
      </tr></thead><tbody>${rows}</tbody>`;
  } else {
    tableTitle.textContent = "Tableau drift par fold";
    const rows = m.folds.map(f => {
      const di = f.drift_info || {};
      const mNo = f.metrics_no_correction;
      const mCo = f.metrics_with_correction;
      const delta = ((mCo.mae_load - mNo.mae_load) * 1000);
      return `<tr>
        <td>Fold ${f.fold}</td>
        <td class="num">${(f.drift_slope_a_per_day || 0).toExponential(2)}</td>
        <td class="num">${fmt(f.drift_intercept_b)}</td>
        <td class="num">${fmt(di.r2 || 0, 3)}</td>
        <td class="num">${di.n_clean_days || di.n_clean || "-"}</td>
        <td class="num">${fmt(mNo.mae_load)}</td>
        <td class="num">${fmt(mCo.mae_load)}</td>
        <td class="num" style="color:${delta < 0 ? 'var(--color-model)' : delta > 0 ? 'var(--color-error)' : 'var(--text-muted)'}">${delta > 0 ? "+" : ""}${delta.toFixed(2)} e-3</td>
      </tr>`;
    }).join("");
    tblEl.innerHTML = `
      <thead><tr>
        <th>Fold</th><th>Pente a (/jour)</th><th>Intercept b</th>
        <th>R\u00B2 fit</th><th>n jours</th>
        <th>MAE sans corr</th><th>MAE avec corr</th><th>Delta MAE</th>
      </tr></thead><tbody>${rows}</tbody>`;
  }

  // ----- Plots par fold -----
  const plot1Title = document.getElementById("drift-plot1-title");
  const plot2Title = document.getElementById("drift-plot2-title");

  if (isV5) {
    plot1Title.textContent = "\u03B3 estime par fold";
    plot2Title.textContent = "R\u00B2 OLS du fit \u03B3 par fold";

    // Bug fix V5 : f.gamma (et non f.gamma_residual)
    const gammas = m.folds.map(f => f.gamma || 0);
    Plotly.newPlot("plot-drift-slope", [
      { x: m.folds.map(f => "Fold " + f.fold), y: gammas, type: "bar",
        marker: { color: gammas.map(g => g < 0 ? colorModel
                                                : getCssVar("--color-error")) },
        text: gammas.map(g => g.toExponential(2)),
        textposition: "outside" },
    ], plotlyLayout({
      yaxis: Object.assign(plotlyLayout().yaxis, {
        title: { text: "\u03B3 (effet PV sur load)", font: { size: 11 } },
        zeroline: true }),
      showlegend: false,
    }), PLOTLY_CONFIG);

    const r2s = m.folds.map(f => (f.gamma_info || {}).r2 || 0);
    Plotly.newPlot("plot-drift-r2", [
      { x: m.folds.map(f => "Fold " + f.fold), y: r2s, type: "bar",
        marker: { color: r2s.map(r => r > 0.10 ? colorModel
                                              : getCssVar("--color-error")) },
        text: r2s.map(r => r.toFixed(3)), textposition: "outside" },
    ], plotlyLayout({
      yaxis: Object.assign(plotlyLayout().yaxis, {
        title: { text: "R\u00B2 OLS load ~ \u03B3\u00B7pv_proxy", font: { size: 11 } },
        range: [0, Math.max(0.5, ...r2s) * 1.2] }),
      showlegend: false,
    }), PLOTLY_CONFIG);
  } else if (isV4) {
    plot1Title.textContent = "\u03B2 estime par fold (correction PV)";
    plot2Title.textContent = "R\u00B2 OLS du fit erreur ~ \u03B2\u00B7pv_proxy par fold";

    const betas = m.folds.map(f => f.pv_correction_beta || 0);
    Plotly.newPlot("plot-drift-slope", [
      { x: m.folds.map(f => "Fold " + f.fold), y: betas, type: "bar",
        marker: { color: betas.map(b => b < 0 ? colorModel
                                              : getCssVar("--color-error")) },
        text: betas.map(b => b.toExponential(2)),
        textposition: "outside" },
    ], plotlyLayout({
      yaxis: Object.assign(plotlyLayout().yaxis, {
        title: { text: "\u03B2 (load std / unite pv_proxy)",
                 font: { size: 11 } },
        zeroline: true }),
      showlegend: false,
    }), PLOTLY_CONFIG);

    const r2s = m.folds.map(f => (f.drift_info || {}).r2 || 0);
    Plotly.newPlot("plot-drift-r2", [
      { x: m.folds.map(f => "Fold " + f.fold), y: r2s, type: "bar",
        marker: { color: r2s.map(r => r > 0.05 ? colorModel
                                              : getCssVar("--text-muted")) },
        text: r2s.map(r => r.toFixed(3)), textposition: "outside" },
    ], plotlyLayout({
      yaxis: Object.assign(plotlyLayout().yaxis, {
        title: { text: "R\u00B2 OLS erreur ~ \u03B2\u00B7pv_proxy", font: { size: 11 } },
        range: [0, Math.max(0.3, ...r2s) * 1.2] }),
      showlegend: false,
    }), PLOTLY_CONFIG);
  } else {
    plot1Title.textContent = "Pente estimee par fold";
    plot2Title.textContent = "R\u00B2 du fit lineaire par fold";

    const slopes = m.folds.map(f => f.drift_slope_a_per_day || 0);
    Plotly.newPlot("plot-drift-slope", [
      { x: m.folds.map(f => "Fold " + f.fold), y: slopes, type: "bar",
        marker: { color: slopes.map(s => s > 0 ? colorModel
                                                : getCssVar("--color-baseline")) },
        text: slopes.map(s => s.toExponential(2)),
        textposition: "outside" },
    ], plotlyLayout({
      yaxis: Object.assign(plotlyLayout().yaxis, {
        title: { text: "Pente / jour", font: { size: 11 } },
        zeroline: true }),
      showlegend: false,
    }), PLOTLY_CONFIG);

    const r2s = m.folds.map(f => (f.drift_info || {}).r2 || 0);
    Plotly.newPlot("plot-drift-r2", [
      { x: m.folds.map(f => "Fold " + f.fold), y: r2s, type: "bar",
        marker: { color: r2s.map(r => r > 0.1 ? colorModel
                                              : getCssVar("--text-muted")) },
        text: r2s.map(r => r.toFixed(3)), textposition: "outside" },
    ], plotlyLayout({
      yaxis: Object.assign(plotlyLayout().yaxis, {
        title: { text: "R\u00B2 du fit lineaire", font: { size: 11 } },
        range: [0, 0.3] }),
      showlegend: false,
    }), PLOTLY_CONFIG);
  }
}

// ============ FEATURES ============
function renderFeatures() {
  const m = M();
  if (!m) return;
  const fi = m.feature_importance || [];
  if (!fi.length) return;
  const sorted = [...fi].sort((a, b) => b.score - a.score);

  const cats = [...new Set(sorted.map(f => f.category))];
  const palette = ["#0F766E", "#1E40AF", "#7C3AED", "#B45309", "#BE123C",
                   "#0891B2", "#65A30D", "#DC2626", "#9333EA", "#0D9488"];
  const catColor = {};
  cats.forEach((c, i) => catColor[c] = palette[i % palette.length]);

  Plotly.newPlot("plot-fi-bar", [{
    x: sorted.map(f => f.score).reverse(),
    y: sorted.map(f => f.label).reverse(),
    customdata: sorted.map(f => f.name).reverse(),
    type: "bar", orientation: "h",
    marker: { color: sorted.map(f => catColor[f.category]).reverse() },
    text: sorted.map(f => f.category).reverse(),
    textposition: "outside",
    hovertemplate: "<b>%{y}</b>" +
                   "<br><span style='color:#94a3b8'>%{customdata}</span>" +
                   "<br>Importance : %{x:.2f}" +
                   "<br>Categorie : %{text}<extra></extra>",
  }], plotlyLayout({
    margin: { l: 320, r: 130, t: 12, b: 40 },
    xaxis: Object.assign(plotlyLayout().xaxis, {
      title: { text: "Importance (gain)", font: { size: 11 } } }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      tickfont: { size: 10, family: "'IBM Plex Sans', sans-serif" },
      automargin: true }),
    showlegend: false,
  }), PLOTLY_CONFIG);

  const agg = {};
  sorted.forEach(f => {
    if (!agg[f.category]) agg[f.category] = { sum: 0, count: 0 };
    agg[f.category].sum += f.score;
    agg[f.category].count++;
  });
  const aggArr = Object.entries(agg).map(([k, v]) => ({
    category: k, sum: v.sum, count: v.count, mean: v.sum / v.count,
  })).sort((a, b) => b.sum - a.sum);

  Plotly.newPlot("plot-fi-category", [{
    x: aggArr.map(a => a.category),
    y: aggArr.map(a => a.sum),
    type: "bar",
    marker: { color: aggArr.map(a => catColor[a.category]) },
    text: aggArr.map(a => a.count + " feat"),
    textposition: "outside",
    hovertemplate: "<b>%{x}</b><br>Importance totale: %{y:.2f}<br>%{text}<extra></extra>",
  }], plotlyLayout({
    margin: { l: 50, r: 18, t: 12, b: 90 },
    xaxis: Object.assign(plotlyLayout().xaxis, {
      tickangle: -25, tickfont: { size: 10 } }),
    yaxis: Object.assign(plotlyLayout().yaxis, {
      title: { text: "Importance totale (gain)", font: { size: 11 } } }),
    showlegend: false,
  }), PLOTLY_CONFIG);

  document.getElementById("tbl-fi-category").innerHTML = `
    <thead><tr>
      <th>Categorie</th><th>n features (top 30)</th>
      <th>Importance totale</th><th>Importance moyenne</th>
    </tr></thead><tbody>
    ${aggArr.map(a => `<tr>
      <td>${a.category}</td>
      <td class="num">${a.count}</td>
      <td class="num">${a.sum.toFixed(2)}</td>
      <td class="num">${a.mean.toFixed(2)}</td>
    </tr>`).join("")}
    </tbody>`;
}

// ============ ANTI-LEAK (frise temporelle supprimee) ============
const STATUS_LABELS = {
  safe: "Lag safe",
  deterministic: "Deterministe",
  forecast: "Prevision pour cible",
  leak_suspect: "Suspect (lag < 192)",
  unknown: "Inconnu",
};
const STATUS_COLORS = {
  safe: "hsl(140, 60%, 45%)",
  deterministic: "hsl(220, 60%, 55%)",
  forecast: "hsl(180, 55%, 45%)",
  leak_suspect: "hsl(0, 70%, 55%)",
  unknown: "hsl(40, 50%, 55%)",
};

function renderLeak() {
  const m = M();
  if (!m) return;
  const leak = m.leak;
  const validation = leak.validation_pipeline;
  const counts = leak.status_counts_top30;
  const isOk = validation.n_warnings === 0
               && (counts.leak_suspect || 0) === 0;

  const banner = isOk
    ? `<div class="leak-status-banner ok">
         <div class="leak-status-icon">OK</div>
         <div class="leak-status-text">
           <strong>Aucun leak detecte pour ${MMeta().display_name}</strong>
           Le pipeline a valide ${m.meta.n_features} features et tous les
           lags respectent la regle minimum de
           ${leak.rule_summary.lag_min_safe_steps} pas.
         </div>
       </div>`
    : `<div class="leak-status-banner warn">
         <div class="leak-status-icon">!</div>
         <div class="leak-status-text">
           <strong>${validation.n_warnings} warning(s) detectes par le pipeline Python</strong>
           ${counts.leak_suspect || 0} feature(s) du top 30 ont un statut suspect.
         </div>
       </div>`;
  document.getElementById("leak-status-banner-wrap").innerHTML = banner;

  const rule = leak.rule_summary;
  document.getElementById("leak-rule").innerHTML = `
    <strong>Convention temporelle</strong> : run du modele a ${rule.run_time},
    cible <code>T &isin; ${rule.target_horizon}</code>.
    <br><br>
    <strong>Regle</strong> : pour toute feature de type
    <code>lag_X_N</code>, le decalage <code>N</code> doit etre &ge;
    <code>${rule.lag_min_safe_steps}</code> pas de 15 min (= 2 jours),
    sinon la donnee correspondante n'existe pas encore au moment du run.
    <br><br>
    <strong>Demonstration</strong> : ${rule.lag_min_safe_explanation}
    <br><br>
    <strong>Cutoff load</strong> : ${rule.load_cutoff || "-"}
    <br>
    <strong>Cutoff meteo</strong> : ${rule.meteo_cutoff || "-"}
  `;

  // Frise temporelle supprimee : seul le camembert des statuts est affiche
  renderLeakStatusPie(counts);

  let validationHtml = `<p><strong>${validation.n_warnings}</strong>
    warning(s) remonte(s) par <code>validate_no_leak()</code> dans
    <code>features.py</code>.</p>`;
  if (validation.warnings && validation.warnings.length > 0) {
    validationHtml += `<ul style="margin-top:8px; padding-left:18px;
      font-family: 'JetBrains Mono', monospace; font-size:11px;">`;
    validation.warnings.forEach(w => {
      validationHtml += `<li style="color: var(--color-error);">${w}</li>`;
    });
    validationHtml += `</ul>`;
  } else {
    validationHtml += `<p style="color: var(--color-model); margin-top:8px;
      font-weight:600;">\u2713 Toutes les colonnes <code>lag_*</code> respectent
      le seuil minimum de ${validation.lag_min_steps} pas.</p>
      <p style="margin-top:8px;">\u2713 La cible <code>load_residual</code> est
      presente dans le dataset.</p>
      <p style="margin-top:8px;">\u2713 Aucune feature calendaire n'a de valeur
      manquante.</p>`;
  }
  document.getElementById("leak-validation-output").innerHTML = validationHtml;

  const fi = m.feature_importance || [];
  const sorted = [...fi].sort((a, b) => b.score - a.score);
  const auditRows = sorted.map((f, idx) => {
    const a = f.audit;
    const lat = a.latest_days;
    const latestStr = lat === null || lat === undefined
      ? "-" : (lat >= 0 ? "+" : "") + lat.toFixed(2) + " j";
    return `<tr>
      <td class="num" style="color: var(--text-faint);">#${idx + 1}</td>
      <td><span style="font-family: 'JetBrains Mono', monospace; font-size:11px;
                color: var(--text-muted);">${f.name}</span><br>${f.label}</td>
      <td>${a.source_label}</td>
      <td class="num">${latestStr}</td>
      <td><span class="status-pill status-${a.status}">${STATUS_LABELS[a.status] || a.status}</span></td>
    </tr>`;
  }).join("");
  document.getElementById("tbl-leak-features").innerHTML = `
    <thead><tr>
      <th>#</th><th>Feature</th><th>Source de la donnee</th>
      <th>Donnee la + recente (rel. au run)</th><th>Statut</th>
    </tr></thead><tbody>${auditRows}</tbody>`;
}

function renderLeakStatusPie(counts) {
  const labels = [], values = [], colors = [];
  Object.entries(counts).forEach(([k, v]) => {
    if (v > 0) {
      labels.push(STATUS_LABELS[k] || k);
      values.push(v);
      colors.push(STATUS_COLORS[k] || "#888");
    }
  });
  Plotly.newPlot("plot-leak-status", [{
    type: "pie", labels: labels, values: values,
    marker: { colors: colors },
    textinfo: "label+value", textposition: "outside",
    hole: 0.5, sort: false,
  }], plotlyLayout({
    margin: { l: 20, r: 20, t: 20, b: 20 },
    showlegend: false,
  }), PLOTLY_CONFIG);
}

// ============ Render orchestrator ============
function renderAll() {
  const active = document.querySelector(".panel.active").id;
  if (active === "panel-synthese") renderSynthese();
  else if (active === "panel-mois") {
    renderCalendar();
    renderMonthlyPlots();
  }
  else if (active === "panel-jour") {
    populateDaySelect();
    renderDay(document.getElementById("day-select").value);
  }
  else if (active === "panel-diagnostics") renderDiagnostics();
  else if (active === "panel-drift") renderDrift();
  else if (active === "panel-features") renderFeatures();
  else if (active === "panel-leak") renderLeak();
}

// ============ Init ============
applyModelColor();
initModelSwitcher();
updateHeaderAndTags();
populateDaySelect();
renderSynthese();

// Resize
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderAll, 200);
});
</script>"""


# ============================================================
# Render + Main
# ============================================================

def render_html(payload: dict) -> str:
    """Injecte les couleurs et donnees dans le template."""
    default_model = payload["default_model"]

    if default_model and default_model in payload["model_meta"]:
        color_model_default = payload["model_meta"][default_model][
            "color_light"]
    else:
        color_model_default = "#047857"

    html = HTML_TEMPLATE
    html = html.replace("__COLOR_REAL__", PALETTE["real"])
    html = html.replace("__COLOR_BASELINE__", PALETTE["baseline"])
    html = html.replace("__COLOR_CORRECTED__", PALETTE["corrected"])
    html = html.replace("__COLOR_ERROR__", PALETTE["error"])
    html = html.replace("__COLOR_ACCENT__", PALETTE["accent"])
    html = html.replace("__COLOR_IRRADIANCE__", PALETTE["irradiance"])
    html = html.replace("__COLOR_MODEL_DEFAULT__", color_model_default)
    html = html.replace("__DEFAULT_MODEL__", default_model or "original")
    html = html.replace("__JS_BLOCK__", JS_BLOCK)

    payload_json = (json.dumps(payload, ensure_ascii=False, default=str)
                    .replace("</", "<\\/"))
    html = html.replace("__DATA_JSON__", payload_json)
    return html


def main(argv: list[str] | None = None) -> int:
    print("=" * 60)
    print("DASHBOARD - generation HTML multi-modeles")
    print("=" * 60)

    print(f"\nChargement des modeles :")
    loaded = {}
    for model_key, cfg in MODEL_CONFIGS.items():
        ds = cfg["dataset"]
        feats_irr = load_irradiance_features(ds)
        if feats_irr is None:
            print(f"  [WARN] {cfg['display_name']} : features parquet "
                  f"absent -> trace irradiance desactivee pour ce modele")
        else:
            print(f"  [info] {cfg['display_name']} : irradiance chargee "
                  f"({feats_irr.shape[0]:,} lignes)")
        data = load_model_data(model_key, feats_irr)
        if data is not None:
            loaded[model_key] = data

    if not loaded:
        print(f"\nERREUR : aucun modele charge.")
        print(f"Lance d'abord les modeles :")
        print(f"  python -m src.model_XGBoost --all")
        print(f"ou individuellement :")
        for key in MODEL_CONFIGS.keys():
            print(f"  python -m src.model_XGBoost --dataset {key}")
        return 1

    print(f"\n{len(loaded)} modele(s) charge(s) : "
          f"{', '.join(MODEL_CONFIGS[k]['display_name'] for k in loaded)}")

    print(f"\nConstruction du payload combine...")
    payload = build_combined_payload(loaded)
    for key in payload["available_models"]:
        m = payload["models"][key]
        print(f"  {MODEL_CONFIGS[key]['display_name']:18s} : "
              f"{m['meta']['n_days']} jours, "
              f"{len(m['daily']['date'])} aggregats journaliers, "
              f"anti-leak {m['leak']['validation_pipeline']['n_warnings']} warnings")

    print(f"\nGeneration HTML...")
    html = render_html(payload)
    size_mb = len(html.encode("utf-8")) / 1024 / 1024

    out_path = DATA_REPORTS / "dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  -> {out_path}")
    print(f"  Taille : {size_mb:.2f} MB")
    print(f"  Modele par defaut a l'ouverture : "
          f"{MODEL_CONFIGS[payload['default_model']]['display_name']}")
    if len(payload["available_models"]) > 1:
        print(f"  Toggle disponible dans l'en-tete pour basculer entre :")
        for key in payload["available_models"]:
            print(f"    - {MODEL_CONFIGS[key]['display_name']}")
    print("\nDashboard genere. Ouvre le fichier HTML dans ton navigateur.")
    return 0


if __name__ == "__main__":
    sys.exit(main())