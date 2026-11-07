"""Configuration module for C-MAPSS predictive maintenance project."""
import yaml
import os

def load_config(path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", path) if not os.path.isabs(path) else path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Default paths - resolved at runtime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "CMAPSSData")
DATA_INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGS_DIR = os.path.join(REPORTS_DIR, "figures")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")

# Ensure directories exist
for d in [DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGS_DIR, DOCS_DIR, CONFIGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Target configuration
TARGET_H: int = 30
TARGET_H_TEST: int = 24

# Feature engineering
LAGS: list[int] = [1, 3, 6, 12, 24]
ROLL_WINDOWS: list[int] = [3, 12, 24]
EWMA_ALPHA: float = 0.3

# Model configuration
TEST_SPLIT_RATIO: float = 0.2
RANDOM_SEED: int = 42

# FFT configuration
FFT_FS: int = 1