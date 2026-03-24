from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

INFLUXDB_URL = "https://timeseries.hevs.ch"
INFLUXDB_TOKEN = (
    "ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0Y"
    "mVrk7hZGPpvb_5aaA-ZxhIw=="
)
INFLUXDB_ORG = "HESSOVS"
INFLUXDB_BUCKET = "MeteoSuisse"

SITES = ["Sion", "Visp", "Montana", "Evolène / Villa"]

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

PRED_VARIABLES = ["PRED_T_2M", "PRED_GLOB", "PRED_DURSUN", "PRED_TOT_PREC",
                   "PRED_RELHUM_2M", "PRED_FF_10M", "PRED_DD_10M", "PRED_PS"]
PRED_VARIANTS = ["ctrl", "q10", "q90", "stde"]

MEASUREMENTS_PRED = [
    f"{var}_{variant}"
    for var in PRED_VARIABLES
    for variant in PRED_VARIANTS
]

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

OIKEN_START = "2022-10-01"
OIKEN_END = "2025-09-30"

TIMEZONE = "Europe/Zurich"