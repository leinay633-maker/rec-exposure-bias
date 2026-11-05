"""Paths, dataset constants and the fixed time windows used throughout the study."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("RECEVAL_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / ".data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
RUNS = ROOT / ".runs"

PURE_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
PURE_MD5 = "0820331067a3784d9691136f772b35a7"
PURE_ARCHIVE = RAW / "KuaiRand-Pure.tar.gz"
PURE_DIR = RAW / "KuaiRand-Pure" / "data"

LOG_FILES = {
    "standard_early": "log_standard_4_08_to_4_21_pure.csv",
    "standard_late": "log_standard_4_22_to_5_08_pure.csv",
    "random": "log_random_4_22_to_5_08_pure.csv",
}
SIDE_FILES = {
    "users": "user_features_pure.csv",
    "videos": "video_features_basic_pure.csv",
}
# Monthly averages that overlap the test period; never loaded (see AGENTS.md).
EXCLUDED_FILES = ("video_features_statistic_pure.csv",)

# Totals published in the KuaiRand README, checked by the audit.
README_TOTALS = {
    "users": 27_285,
    "standard_interactions": 1_436_609,
    "standard_items": 7_551,
    "random_interactions": 1_186_059,
    "random_items": 7_583,
}

# Inclusive date windows (yyyymmdd ints, as stored in the logs' `date` column).
TRAIN = (20220408, 20220421)
VALID = (20220422, 20220428)
CALIB = (20220422, 20220428)
TEST = (20220429, 20220508)
