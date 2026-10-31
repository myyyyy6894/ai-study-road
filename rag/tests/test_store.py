"""向量存储的单元测试（不依赖外部服务）。"""

import numpy as np
import pytest

from app.retrieval.store import VectorStore

DIM = 8


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _make_store() -> tuple[VectorStore, np.ndarray]:
    rng = np.random.default_rng(0)
    vectors = _unit(rng.normal(size=(3, DIM)).astype(np.float32))
    chunks = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": f"text{i}"} for i in range(3)]

    store = VectorStore(dim=DIM)
    store.add(vectors, chunks)
    return store, vectors


class TestAdd:
    def test_len(self):
        store, _ = _make_store()
        assert len(store) == 3
        assert store.vectors.shape == (3, DIM)

    def test_dimension_mismatch_raises(self):
        store = VectorStore(dim=DIM)
        with pytest.raises(ValueError):
            store.add(np.zeros((2, DIM + 1), dtype=np.float32), [{}, {}])

    def test_count_mismatch_raises(self):
        store = VectorStore(dim=DIM)
        with pytest.raises(ValueError):
            store.add(np.zeros((2, DIM), dtype=np.float32), [{}])


class TestSearch:
    def test_self_query_ranks_first(self):
        store, vectors = _make_store()
        results = store.search(vectors[1], top_k=1)
        assert results[0][0]["chunk_id"] == "c1"
        assert results[0][1] == pytest.approx(1.0)

    def test_top_k_limit(self):
        store, vectors = _make_store()
        assert len(store.search(vectors[0], top_k=2)) == 2

    def test_empty_store_returns_empty(self):
        assert VectorStore(dim=DIM).search(np.zeros(DIM, dtype=np.float32), top_k=3) == []

    def test_scores_descending(self):
        store, vectors = _make_store()
        scores = [s for _, s in store.search(vectors[0], top_k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_accepts_1d_and_2d_query(self):
        store, vectors = _make_store()
        one_d = store.search(vectors[0], top_k=1)[0][0]["chunk_id"]
        two_d = store.search(vectors[0].reshape(1, -1), top_k=1)[0][0]["chunk_id"]
        assert one_d == two_d


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        store, vectors = _make_store()
        store.save(tmp_path)

        loaded = VectorStore()
        loaded.load(tmp_path)

        assert len(loaded) == 3
        assert loaded.dim == DIM
        assert loaded.search(vectors[0], top_k=1)[0][0]["chunk_id"] == "c0"


class TestRemoveDoc:
    def test_remove_existing(self):
        store, _ = _make_store()
        removed = store.remove_doc("d1")
        assert removed == 3
        assert len(store) == 0

    def test_remove_missing_returns_zero(self):
        store, _ = _make_store()
        assert store.remove_doc("nope") == 0
        assert len(store) == 3
