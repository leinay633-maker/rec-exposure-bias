"""Paths and fixed settings of study 3 (docs/rlrec/PROTOCOL.md)."""
from __future__ import annotations

from receval import config as rc

ROOT = rc.ROOT
RUNS = ROOT / ".runs" / "rlrec"
MODELS = RUNS / "models"
TEXT_DIR = rc.DATA / "kuairand_text"
CAPTIONS = TEXT_DIR / "kuairand_video_captions.csv"
CATEGORIES = TEXT_DIR / "kuairand_video_categories.csv"
TEXT_MD5 = {"kuairand_video_captions.csv": "9fd4e7587ffd0c8dc289d013858fb229",
            "kuairand_video_categories.csv": "8a4c772d3d8f8cf65fa1e1eb896f4177"}

TRAIN, VALID, TEST = rc.TRAIN, rc.VALID, rc.TEST
FIT, HOLD = (20220422, 20220425), (20220426, 20220428)
REF_DATE = 20220422  # the reference context's date (§5)

SEEDS = (0, 1, 2)
HISTORY = 50
K_MAIN, KS = 500, (100, 200, 500, 1000)
KL_LEVELS, KL_MAIN = (0.5, 1.0, 2.0), 1.0
KL_SUBSET, KL_SUBSET_SEED = 2_000, 2  # §6.1
VALID_SUBSET, VALID_SUBSET_SEED = 2_000, 1  # §4
TEST_ROWS = 892_896  # random test tab-1 rows (FEASIBILITY.md); §9 scales it by the HOLD row retention

PROTOCOL_FREEZE = "76d02f5"  # the commit that froze docs/rlrec/PROTOCOL.md v2 (§0, §10)
