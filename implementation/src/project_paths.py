from pathlib import Path

IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = IMPLEMENTATION_ROOT.parent

DATA_DIR = IMPLEMENTATION_ROOT / "data"
CONFIG_DIR = IMPLEMENTATION_ROOT / "config"
ARTIFACTS_MODELS_DIR = IMPLEMENTATION_ROOT / "artifacts" / "models"
ARTIFACTS_DATA_DIR = IMPLEMENTATION_ROOT / "artifacts" / "data"
REPORTS_FIGURES_DIR = IMPLEMENTATION_ROOT / "reports" / "figures"

for d in [DATA_DIR, CONFIG_DIR, ARTIFACTS_MODELS_DIR, ARTIFACTS_DATA_DIR, REPORTS_FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def input_file(name: str) -> Path:
    """Resolve an input file from common old/new locations."""
    candidates = [
        DATA_DIR / name,
        IMPLEMENTATION_ROOT / name,
        REPO_ROOT / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find '{name}'. Looked in: " + ", ".join(str(c) for c in candidates)
    )


def model_file(name: str) -> Path:
    return ARTIFACTS_MODELS_DIR / name


def artifact_data_file(name: str) -> Path:
    return ARTIFACTS_DATA_DIR / name


def config_file(name: str) -> Path:
    return CONFIG_DIR / name


def figure_file(name: str) -> Path:
    return REPORTS_FIGURES_DIR / name
