import numpy as np
import pytest
import torch

from receval import features as F
from receval.models import RANKERS, TwoTowerRetriever
from receval.retrieval import recall_ndcg


def _inputs(batch=8):
    cards = [5] * len(F.CAT_FIELDS)
    gen = torch.Generator().manual_seed(0)
    cat = torch.randint(0, 5, (batch, len(F.CAT_FIELDS)), generator=gen)
    num = torch.randn(batch, 9, generator=gen)
    hist = torch.randint(0, 5, (batch, F.HISTORY), generator=gen)
    return cards, cat, num, hist


@pytest.mark.parametrize("name", sorted(RANKERS))
def test_rankers_return_one_logit_per_row(name):
    cards, cat, num, hist = _inputs()
    space = F.FeatureSpace()
    model = RANKERS[name](cards, 9, numeric_names=space.numeric_names, dim=4)
    out = model(cat, num, hist)
    assert out.shape == ((8, 3) if name == "mmoe" else (8,))
    assert torch.isfinite(out).all()


def test_retriever_towers_are_unit_norm():
    cards, cat, _, hist = _inputs()
    model = TwoTowerRetriever(cards, dim=4, out=8)
    assert torch.allclose(model.users(cat, hist).norm(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(model.items(cat).norm(dim=-1), torch.ones(8), atol=1e-5)


class _Pool:
    def __init__(self, seen):
        self.seen = seen


def test_recall_excludes_seen_items_and_counts_hits():
    scores = torch.tensor([[9.0, 8.0, 7.0, 1.0] + [0.0] * 196,
                           [1.0, 2.0, 3.0, 4.0] + [0.0] * 196])
    pool = _Pool({10: np.array([0])})  # user 10 already saw item 0, so item 2 enters the top two
    targets = {10: np.array([2]), 11: np.array([3, 199])}
    table = recall_ndcg(scores, [10, 11], pool, targets, ks=(2, 100, 200))
    assert table.loc[10, "recall@2"] == 1.0
    assert table.loc[11, "recall@2"] == 0.5
    assert table.loc[11, "recall@200"] == 1.0
    assert 0 < table.loc[11, "ndcg@100"] <= 1
