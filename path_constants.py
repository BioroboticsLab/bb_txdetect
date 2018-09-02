"""default locations for generated and required files"""
from pathlib import Path

_BASE_FOLDER = Path().home() / "txdetect_data"
_IMAGES = Path(_BASE_FOLDER) / "images"

ARCHIVE_PATH = _BASE_FOLDER / "saved_models"
if not ARCHIVE_PATH.exists():
    ARCHIVE_PATH.mkdir()
ARCHIVE_PATH = str(ARCHIVE_PATH)


# locations of files during training.
# after successful training they get moved to a subfolder in ARCHIVE_PATH.
MODEL_PATH = str(_BASE_FOLDER / "saved_model")
TRAIN_STATS = str(_BASE_FOLDER / "train_stats.csv")
TRAIN_LOG = str(_BASE_FOLDER / "trainlog.txt")
DEBUG_LOG = str(_BASE_FOLDER / "debug.txt")
PARAMETERS_JSON = str(_BASE_FOLDER / "parameters.json")

CLAHE = "_clahe"
INVERT = "_invert"
PAD = "_pad_"

LABEL_YES = "y"
LABEL_NO = "n"
LABEL_UNKNOWN = "u"

# default image folder for training
IMG_FOLDER = str(_IMAGES / ("images" + PAD + "16"))

# default location for loading candidates for validation
CANDIDATES_JSON_PATH = \
    "/mnt/storage/david/cache/beesbook/trophallaxis/candidate_events.json"
# output folder for candidates
CANDIDATES_IMAGE_FOLDER_RAW = str(_IMAGES / "candidates")
