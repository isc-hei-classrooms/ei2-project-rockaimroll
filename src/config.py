"""
Configuration centralisée du projet Energy Informatics 2.

Tous les paramètres du projet sont regroupés ici pour éviter
de les répéter dans chaque fichier. Si on veut changer un site,
une période ou une connexion, on modifie uniquement ce fichier.
"""

from pathlib import Path

# ============================================================
# Chemins du projet
# ============================================================
# Path(__file__) = chemin de CE fichier (config.py)
# .resolve() = chemin absolu (pas relatif)
# .parent.parent = on remonte 2 niveaux : src/ → racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dossiers où on stocke les données
DATA_RAW = PROJECT_ROOT / "data" / "raw"            # Données brutes (CSV, Parquet d'acquisition)
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" # Données nettoyées et fusionnées

# ============================================================
# Connexion InfluxDB (base de données temporelles de la HES-SO)
# ============================================================
# Le serveur timeseries.hevs.ch héberge les données MeteoSuisse.
# Le token est partagé par la classe pour accéder au bucket du cours.
INFLUXDB_URL = "https://timeseries.hevs.ch"
INFLUXDB_TOKEN = (
    "ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0Y"
    "mVrk7hZGPpvb_5aaA-ZxhIw=="
)
INFLUXDB_ORG = "HESSOVS"
INFLUXDB_BUCKET = "MeteoSuisse"

# ============================================================
# Sites météo retenus
# ============================================================
# On prend 4 stations qui couvrent le périmètre géographique d'OIKEN
# (Valais central), à différentes altitudes et expositions.
# Plus tard dans la normalisation, on fera la moyenne des 4 sites.
SITES = ["Sion", "Visp", "Montana", "Evolène / Villa"]

# ============================================================
# Mesures réelles (données observées toutes les 10 minutes)
# ============================================================
# Ce sont les 9 variables météo mesurées en temps réel par MeteoSuisse.
# Chaque nom correspond exactement à une "measurement" dans InfluxDB.
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

# ============================================================
# Mesures de prévision (modèle COSMO-E de MétéoSuisse)
# ============================================================
# COSMO-E produit des prévisions probabilistes avec 21 membres d'ensemble.
# On extrait 4 variantes pour chaque variable :
#   - ctrl : run de contrôle (prévision déterministe)
#   - q10  : quantile 10% (scénario bas / optimiste pour la charge)
#   - q90  : quantile 90% (scénario haut / pessimiste)
#   - stde : écart-type de l'ensemble (incertitude de la prévision)
PRED_VARIABLES = ["PRED_T_2M", "PRED_GLOB", "PRED_DURSUN", "PRED_TOT_PREC",
                   "PRED_RELHUM_2M", "PRED_FF_10M", "PRED_DD_10M", "PRED_PS"]
PRED_VARIANTS = ["ctrl", "q10", "q90", "stde"]

# Génération automatique de toutes les combinaisons :
# ex: PRED_T_2M_ctrl, PRED_T_2M_q10, PRED_T_2M_q90, PRED_T_2M_stde, ...
# Ça donne 8 variables × 4 variantes = 32 measurements au total
MEASUREMENTS_PRED = [
    f"{var}_{variant}"
    for var in PRED_VARIABLES
    for variant in PRED_VARIANTS
]

# ============================================================
# Noms courts pour les colonnes (lisibilité dans les DataFrames)
# ============================================================
# Les noms InfluxDB sont très longs. On les remplace par des noms
# courts et clairs pour travailler plus facilement dans le code.
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

# ============================================================
# Paramètres temporels
# ============================================================
# Période couverte par le dataset OIKEN (3 ans de données)
OIKEN_START = "2022-10-01"
OIKEN_END = "2025-09-30"

# Fuseau horaire local (pour la gestion heure d'été / heure d'hiver)
TIMEZONE = "Europe/Zurich"