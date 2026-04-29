"""
Configuration centralisee du projet OIKEN ML.

Toute valeur partagee entre plusieurs modules vit ici. Aucune logique,
uniquement des constantes. Les modules font `from src.config import ...`.

Sections:
  - Chemins et arborescence
  - Connexion InfluxDB
  - Sites meteo (perimetre OIKEN)
  - Mesures InfluxDB (reelles + previsionnelles)
  - Bornes physiques pour la detection d'outliers
  - Periode et fuseau OIKEN
"""

from pathlib import Path

# ============================================================
# 1. Chemins
# ============================================================
# PROJECT_ROOT pointe sur la racine du repo, peu importe d'ou le
# script est appele. .parent.parent suppose que ce fichier est en
# src/config.py. Adapter si reorganisation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"            # sorties d'acquisition
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"  # sorties normalisees
DATA_REPORTS = PROJECT_ROOT / "data" / "reports"      # rapports qualite

# Le CSV source OIKEN doit etre depose ici avant l'execution.
# Le nom exact depend du dataset choisi (cf section 7) :
#   data/raw/oikendata_original.csv   (dataset "original")
#   data/raw/oikendata_golden.csv     (dataset "golden")
# La constante OIKEN_CSV_PATH (legacy) pointe sur le dataset par defaut
# et est definie a la fin de la section 7.


# ============================================================
# 2. InfluxDB (HES-SO)
# ============================================================
# Token partage du cours. Si rotation de cle, mettre a jour ici uniquement.
INFLUXDB_URL = "https://timeseries.hevs.ch"
INFLUXDB_TOKEN = (
    "ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0Y"
    "mVrk7hZGPpvb_5aaA-ZxhIw=="
)
INFLUXDB_ORG = "HESSOVS"
INFLUXDB_BUCKET = "MeteoSuisse"        # bucket principal (real + pred)
# Note: le bucket "MeteoSuissePrevision" existe mais a ete verifie
# vide via scripts.exploration. On l'ignore.

# Timeout de requete Flux (ms). Les requetes lourdes (3 ans, 16 leads,
# tous les sites) peuvent depasser le defaut de 60s.
INFLUX_TIMEOUT_MS = 600_000  # 10 minutes


# ============================================================
# 3. Sites meteo
# ============================================================
# Choix: couvrir le perimetre OIKEN (Valais central) avec une variete
# d'altitudes et de microclimats. Verifier qu'un site retenu a bien
# des donnees sur toute la periode (verification automatique faite
# dans acquisition_meteo_real.py).
#
# Categorisation:
#   - PLAINE        : Sion (482m), Visp (640m), Bouveret (375m)
#   - COTEAU MOYEN  : Montana (1428m), Evolene/Villa (1825m),
#                     Grachen (1605m), Montagnier Bagnes (820m)
#   - HAUTE MONTAGNE: Mottec (1564m), Zermatt (1638m), Evionnaz (482m)
#
# Note typographique: "Evolene / Villa" contient un ASCII slash + espaces.
# C'est exactement la chaine attendue par InfluxDB (verifie via
# explore_influx.py). Toute modification casse la requete.
SITES_VALAIS_CENTRAL = [
    "Sion",
    "Visp",
    "Montana",
    "Evolène / Villa",
]

# Sites etendus (a activer apres verification de couverture temporelle).
SITES_EXTENDED = SITES_VALAIS_CENTRAL + [
    "Mottec",
    "Grächen",
    "Zermatt",
    "Evionnaz",
    "Bouveret",
    "Montagnier, Bagnes",
]

# SITES = liste effectivement utilisee. On commence avec EXTENDED ;
# l'acquisition retire automatiquement les sites incomplets.
SITES = SITES_EXTENDED


# ============================================================
# 4. Mesures InfluxDB
# ============================================================
# 4.a) Reelles : 9 capteurs MeteoSuisse, pas natif 10 minutes.
MEASUREMENTS_REAL = [
    "Air temperature 2m above ground (current value)",
    "Atmospheric pressure at barometric altitude",
    "Global radiation (ten minutes mean)",
    "Gust peak (one second) (maximum)",
    "Precipitation (ten minutes total)",
    "Relative air humidity 2m above ground (current value)",
    "Sunshine duration (ten minutes total)",
    "Wind Direction (ten minutes mean)",
    "Wind speed scalar (ten minutes mean)",
]

# 4.b) Previsionnelles : COSMO-E, 8 variables * 4 statistiques = 32.
# Les statistiques representent la dispersion de l'ensemble (21 membres):
#   ctrl = run de controle (membre deterministe)
#   q10  = quantile 10% (borne basse de l'ensemble)
#   q90  = quantile 90% (borne haute de l'ensemble)
#   stde = ecart-type (dispersion)
PRED_VARIABLES = [
    "PRED_T_2M",      # Temperature 2m
    "PRED_GLOB",      # Rayonnement global
    "PRED_DURSUN",    # Duree d'ensoleillement
    "PRED_TOT_PREC",  # Precipitations totales
    "PRED_RELHUM_2M", # Humidite relative 2m
    "PRED_FF_10M",    # Vitesse vent 10m
    "PRED_DD_10M",    # Direction vent 10m
    "PRED_PS",        # Pression au sol
]
PRED_VARIANTS = ["ctrl", "q10", "q90", "stde"]

MEASUREMENTS_PRED = [
    f"{var}_{variant}"
    for var in PRED_VARIABLES
    for variant in PRED_VARIANTS
]

# 4.c) SEMANTIQUE DU TAG InfluxDB "Prediction"
# ============================================================
# Le tag "Prediction" est le LEAD TIME : delai (en heures, zero-padde
# sur 2 chiffres) entre l'instant ou la prevision a ete emise et
# l'instant cible "_time". Ce N'EST PAS un identifiant de run de modele.
#
# Exemple verifie sur PRED_GLOB_ctrl @ Sion, _time = 2025-06-15 12:00 UTC :
#   Prediction="01" -> 785.7 W/m2  (prevision faite a 11:00 UTC pour 12:00)
#   Prediction="03" -> 818.4 W/m2  (prevision faite a 09:00 UTC pour 12:00)
#   Prediction="06" -> 297.7 W/m2  (prevision faite a 06:00 UTC pour 12:00)
#   Prediction="12" -> 458.9 W/m2  (prevision faite a 00:00 UTC pour 12:00)
#   Prediction="24" -> 757.5 W/m2  (prevision faite a 12:00 UTC J-1)
#   Prediction="33" -> 742.2 W/m2  (prevision faite a 03:00 UTC J-1)
#
# CAS PARTICULIER - leads "00" sur les variables CUMULEES :
# Pour PRED_GLOB, PRED_DURSUN, PRED_TOT_PREC a Prediction="00", le cumul
# est mathematiquement non defini (rien n'a encore ete cumule a t=0).
# MeteoSuisse encode cela par la sentinelle -99999. C'est l'origine
# du bug initialement diagnostique (cf reports/diag_pred_glob.txt).
# On EXCLUT donc systematiquement Prediction="00" de l'acquisition.
#
# Pour un forecast J+1 publie le matin (run 00 UTC, dispo ~03 UTC),
# les leads couvrant les 24h du jour J+1 sont 22 a 47. En pratique
# COSMO-E publie peu de valeurs au-dela de 33h (verifie sur le diag).
# On retient donc "18" a "33" (16 leads horaires) qui couvre :
#   - run 00 UTC -> J+1 partiel (lead 22-33 sur les 24h du lendemain)
#   - run 06 UTC -> J+1 quasi complet (lead 18-33 sur 16h du lendemain)
# La selection finale d'un lead par instant cible se fait dans
# normalization.py selon la regle metier choisie.

# Liste des leads a recuperer en acquisition. Format : strings zero-padded.
# Modifier ici pour elargir/restreindre la fenetre temporelle d'acquisition.
PRED_LEAD_TIMES: list[str] = [f"{h:02d}" for h in range(18, 34)]  # "18" a "33"

# Lead time par defaut a utiliser en aval (selection unique dans
# normalization.py si l'on ne fait pas de stratagie multi-lead).
# Doit etre present dans PRED_LEAD_TIMES.
PRED_LEAD_TIME_DEFAULT: str = "24"
assert PRED_LEAD_TIME_DEFAULT in PRED_LEAD_TIMES, (
    f"PRED_LEAD_TIME_DEFAULT='{PRED_LEAD_TIME_DEFAULT}' doit etre dans "
    f"PRED_LEAD_TIMES={PRED_LEAD_TIMES}"
)

# DEPRECATED - Garde pour retro-compatibilite avec d'anciens scripts
# qui referenceraient encore cette constante. NE PLUS UTILISER.
# Le tag "Prediction" a la valeur "00" donne :
#   - pour les variables instantanees : l'analyse (etat initial du modele)
#   - pour les variables cumulees : la sentinelle -99999 (bug initial)
PRED_RUN = "00"  # noqa: deprecated


# ============================================================
# 5. Renommage des colonnes
# ============================================================
# Les noms InfluxDB sont longs et fragiles. On les mappe vers des noms
# courts, snake_case, utilisables comme noms de colonnes Polars.
MEASUREMENT_SHORT_NAMES = {
    "Air temperature 2m above ground (current value)": "temperature_2m",
    "Atmospheric pressure at barometric altitude": "pressure",
    "Global radiation (ten minutes mean)": "global_radiation",
    "Gust peak (one second) (maximum)": "gust_peak",
    "Precipitation (ten minutes total)": "precipitation",
    "Relative air humidity 2m above ground (current value)": "humidity",
    "Sunshine duration (ten minutes total)": "sunshine_duration",
    "Wind Direction (ten minutes mean)": "wind_direction",
    "Wind speed scalar (ten minutes mean)": "wind_speed",
}

# Pour les previsions, le nom InfluxDB en minuscules suffit (deja court).


# ============================================================
# 6. Bornes physiques (validation des donnees)
# ============================================================
# Plages plausibles pour une station valaisanne (altitude 300-1900m).
# Toute valeur en dehors est marquee comme outlier en normalisation,
# puis remplacee selon une strategie (NULL + interpolation, voir
# normalization.py). Sources: ranges typiques MeteoSuisse + bon sens.
PHYSICAL_BOUNDS = {
    "temperature_2m":     (-40.0,  45.0),  # °C
    "pressure":           ( 700.0, 1050.0),  # hPa (corrige niveau mer)
    "global_radiation":   (   0.0, 1400.0),  # W/m² (max theorique ~1100)
    "gust_peak":          (   0.0,  60.0),  # m/s (au-dela = tempete)
    "precipitation":      (   0.0,  50.0),  # mm/10min (extreme)
    "humidity":           (   0.0, 100.0),  # %
    "sunshine_duration":  (   0.0,  10.0),  # min/10min
    "wind_direction":     (   0.0, 360.0),  # degres
    "wind_speed":         (   0.0,  40.0),  # m/s moyen
}

# Charge OIKEN: standardisee, donc theoriquement centree autour de 0.
# Pas de borne physique stricte, mais un seuil sigma pour outliers.
LOAD_OUTLIER_SIGMA = 6.0
PV_MIN_KWH = 0.0          # PV ne peut pas etre negatif
PV_MAX_KWH = 50_000.0     # plafond defensif (reseau OIKEN)


# ============================================================
# 7. Datasets et periode
# ============================================================
# Le projet supporte plusieurs datasets OIKEN en parallele : on en
# ajoute autant que necessaire dans la dict DATASETS. Chaque dataset
# decrit un CSV source, sa plage temporelle attendue, et un label
# affichable. Toutes les sorties (parquets et rapports) sont
# automatiquement suffixees par le nom du dataset, ce qui permet de
# faire tourner le pipeline plusieurs fois sans collision.
#
# Convention de nommage des fichiers de sortie :
#   data/raw/oiken_raw_<name>.parquet
#   data/processed/dataset_normalized_<name>.parquet
#   data/processed/dataset_features_<name>.parquet
#   data/processed/predictions_xgboost_cv_<name>.parquet
#   data/models/xgboost_j1_final_<name>.json
#   data/reports/model_xgboost_report_<name>.json
#
# La meteo (real + pred) reste PARTAGEE entre tous les datasets car
# elle decrit les memes capteurs/previsions sur les memes plages
# physiques. Donc pas de suffixe sur meteo_real_raw.parquet et
# meteo_pred_raw.parquet.

DATASETS: dict[str, dict] = {
    "original": {
        # CSV source dans data/raw/. Si tu utilises encore l'ancien nom
        # "oikendata.csv", renomme-le ou ajoute un lien symbolique.
        "csv_path": DATA_RAW / "oikendata_original.csv",
        "expected_rows": 105_120,
        "start": "2022-10-01",
        "end": "2025-09-30",
        "label": "Original (3 ans)",
        # Pas d'override : utilise les defaults de model_XGBoost
        # (N_FOLDS=5, TEST_SIZE_DAYS=60, MIN_TRAIN_DAYS=720).
        # 720 + 5*60 = 1020 jours requis, dispo apres lags ~1064.
    },
    "golden": {
        "csv_path": DATA_RAW / "oikendata_golden.csv",
        "expected_rows": 92_544,
        "start": "2023-09-01",
        "end": "2026-04-22",
        "label": "Golden (2.6 ans)",
        # Le golden est trop court pour la stratégie CV par defaut :
        # apres drop des lignes a lags incomplets (rolling 30j), il
        # reste ~927 jours. Defauts requierent 720+5*60 = 1020 jours.
        # Compromis : on baisse min_train_days a 600 (= 1.6 an), on
        # garde N_FOLDS=5 et TEST_SIZE_DAYS=60 pour COMPARABILITE
        # FOLD-PAR-FOLD avec original (memes fenetres de test 60j).
        # Total requis : 600 + 5*60 = 900 jours, marge ~27 jours.
        "cv_overrides": {
            "min_train_days": 600,
        },
    },
}

# Dataset utilise par defaut si aucun --dataset n'est passe au script.
DEFAULT_DATASET = "original"

# Validation a l'import : detecter immediatement une faute de frappe
# dans DEFAULT_DATASET plutot qu'au runtime.
assert DEFAULT_DATASET in DATASETS, (
    f"DEFAULT_DATASET={DEFAULT_DATASET!r} doit etre une cle de DATASETS "
    f"({list(DATASETS.keys())})"
)


def get_dataset_config(name: str) -> dict:
    """Retourne la config d'un dataset, leve ValueError si inconnu."""
    if name not in DATASETS:
        raise ValueError(
            f"Dataset inconnu : {name!r}. "
            f"Datasets configures : {list(DATASETS.keys())}"
        )
    return DATASETS[name]


def get_oiken_raw_path(dataset: str) -> Path:
    """Chemin du parquet brut OIKEN pour un dataset donne."""
    get_dataset_config(dataset)  # validation
    return DATA_RAW / f"oiken_raw_{dataset}.parquet"


def get_normalized_path(dataset: str) -> Path:
    """Chemin du parquet normalise pour un dataset donne."""
    get_dataset_config(dataset)
    return DATA_PROCESSED / f"dataset_normalized_{dataset}.parquet"


def get_features_path(dataset: str) -> Path:
    """Chemin du parquet features pour un dataset donne."""
    get_dataset_config(dataset)
    return DATA_PROCESSED / f"dataset_features_{dataset}.parquet"


def get_model_path(dataset: str) -> Path:
    """Chemin du modele XGBoost final pour un dataset donne."""
    get_dataset_config(dataset)
    return PROJECT_ROOT / "data" / "models" / f"xgboost_j1_final_{dataset}.json"


def get_predictions_path(dataset: str) -> Path:
    """Chemin des predictions CV pour un dataset donne."""
    get_dataset_config(dataset)
    return DATA_PROCESSED / f"predictions_xgboost_cv_{dataset}.parquet"


def get_model_report_path(dataset: str) -> Path:
    """Chemin du rapport modele JSON pour un dataset donne."""
    get_dataset_config(dataset)
    return DATA_REPORTS / f"model_xgboost_report_{dataset}.json"


# ============================================================
# 7bis. Constantes legacy (retrocompatibilite)
# ============================================================
# Ces constantes pointent vers le DEFAULT_DATASET. Elles existent pour
# que le code legacy continue de tourner sans modification, mais les
# nouveaux scripts doivent utiliser les helpers ci-dessus.
_default = DATASETS[DEFAULT_DATASET]
OIKEN_CSV_PATH = _default["csv_path"]
OIKEN_START = _default["start"]
OIKEN_END = _default["end"]
OIKEN_EXPECTED_ROWS = _default["expected_rows"]

METEO_START = OIKEN_START          # ou une autre date si tu veux décaler
METEO_END   = "2026-04-27"
# Fuseau dans lequel les timestamps OIKEN sont ETIQUETES.
# Hypothese forte (a verifier en normalisation): le CSV est en heure
# locale suisse, avec changements DST. Les timestamps sont des
# "wall-clock" locaux, pas des UTC.
TIMEZONE_OIKEN = "Europe/Zurich"

# Pas de temps cible pour la fusion finale.
TARGET_RESOLUTION = "15m"