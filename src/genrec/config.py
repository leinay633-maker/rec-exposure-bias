"""Study 2 (KuaiRec): paths, archive checksums and the fixed choices written into docs/kuairec/PROTOCOL.md."""
from __future__ import annotations

from receval.config import ROOT

DATA = ROOT / ".data" / "kuairec"
RAW = DATA / "raw"
CACHE = DATA / "cache"
RUNS = ROOT / ".runs" / "kuairec"
DOCS = ROOT / "docs" / "kuairec"

ARCHIVE = DATA / "KuaiRec.zip"
ARCHIVE_URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
ARCHIVE_MD5 = "261550d472c48eff4990fb13c0e5bcf7"
CAPTION_MD5 = "31bc38cdccdf75a71df137779035f8cb"  # kuairec_caption_category.csv, inside the archive too

BIG, SMALL, CAPTION, CATEGORIES = "big_matrix.csv", "small_matrix.csv", "kuairec_caption_category.csv", "item_categories.csv"
# Published totals (KuaiRec README), checked by the audit.
BIG_TOTALS = {"rows": 12_530_806, "users": 7_176, "items": 10_728}
SMALL_TOTALS = {"rows": 4_676_570, "users": 1_411, "items": 3_327}

# Windows on the big matrix (dates inclusive). The matrix has data on 7/05-7/12, 8/01-8/10 and 8/27-9/05 only.
TRAIN = (20200705, 20200810)
VALID = (20200827, 20200831)
NATURAL_TEST = (20200901, 20200905)  # sealed
POSITIVE_RATIO = 2.0  # positive: watch_ratio > 2.0 (KuaiRec paper §3.1.3, README)

SPLIT_SEED = 20261101  # proxy / test users and the biased-test draws
PROXY_SHARE = 0.2
VALID_SUBSET, VALID_SUBSET_SEED = 2000, 1
BIASED_DRAWS = 20
HISTORY = 50
SEEDS = (0, 1, 2)
