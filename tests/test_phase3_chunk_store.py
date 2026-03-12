"""
Phase 3 tests: full chunk text moved out of Qdrant into SQLite chunk store.

Run with:
    python -m pytest tests/test_phase3_chunk_store.py -v
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch


def _fake_embedding(dim: int = 384):
    """Return a mock embedding that supports .tolist() like a numpy array."""
    vec = [0.1] * dim
    m = MagicMock()
    m.tolist.return_value = vec
    return m


class TestChunkStoreFunctions(unittest.TestCase):
    """write_chunks, fetch_chunk_text, delete_chunks_for_doc."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._db = Path(self._tmp.name) / "chunks.sqlite3"
        import labvdb
        self._original = labvdb.CHUNK_STORE_PATH
        labvdb.CHUNK_STORE_PATH = self._db

    def tearDown(self):
        import labvdb
        labvdb.CHUNK_STORE_PATH = self._original
        self._tmp.cleanup()

    def test_write_and_fetch_chunk(self):
        """write_chunks persists; fetch_chunk_text retrieves by chunk_id."""
        import labvdb
        labvdb.write_chunks([("chunk-1", "doc-a", "full text here")])
        result = labvdb.fetch_chunk_text("chunk-1")
        self.assertEqual(result, "full text here")

    def test_fetch_missing_chunk_returns_none(self):
        """fetch_chunk_text returns None for an unknown chunk_id."""
        import labvdb
        labvdb.ensure_chunk_db()
        result = labvdb.fetch_chunk_text("nonexistent")
        self.assertIsNone(result)

    def test_write_chunks_multiple_rows(self):
        """write_chunks inserts all rows in one call."""
        import labvdb
        rows = [
            ("c1", "doc-a", "text one"),
            ("c2", "doc-a", "text two"),
            ("c3", "doc-b", "text three"),
        ]
        labvdb.write_chunks(rows)
        self.assertEqual(labvdb.fetch_chunk_text("c1"), "text one")
        self.assertEqual(labvdb.fetch_chunk_text("c2"), "text two")
        self.assertEqual(labvdb.fetch_chunk_text("c3"), "text three")

    def test_write_chunks_is_idempotent(self):
        """Re-writing the same chunk_id overwrites the previous text."""
        import labvdb
        labvdb.write_chunks([("c1", "doc-a", "original")])
        labvdb.write_chunks([("c1", "doc-a", "updated")])
        self.assertEqual(labvdb.fetch_chunk_text("c1"), "updated")

    def test_delete_chunks_for_doc_removes_all_doc_rows(self):
        """delete_chunks_for_doc removes all rows for that doc_id only."""
        import labvdb
        labvdb.write_chunks([
            ("c1", "doc-a", "text a1"),
            ("c2", "doc-a", "text a2"),
            ("c3", "doc-b", "text b1"),
        ])
        labvdb.delete_chunks_for_doc("doc-a")
        self.assertIsNone(labvdb.fetch_chunk_text("c1"))
        self.assertIsNone(labvdb.fetch_chunk_text("c2"))
        self.assertEqual(labvdb.fetch_chunk_text("c3"), "text b1")  # untouched


class TestFlushRecordsChunkStore(unittest.TestCase):
    """flush_records() writes to chunk store before Qdrant; payload has preview not text."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._db = Path(self._tmp.name) / "chunks.sqlite3"
        import labvdb
        self._original = labvdb.CHUNK_STORE_PATH
        labvdb.CHUNK_STORE_PATH = self._db

    def tearDown(self):
        import labvdb
        labvdb.CHUNK_STORE_PATH = self._original
        self._tmp.cleanup()

    def test_qdrant_payload_has_preview_not_text(self):
        """After flush_records, upserted points must have 'preview' and no 'text' field."""
        from indexer import flush_records

        full_text = "A" * 500
        pending = [(1, 0, "Methods", full_text)]
        metadata = {
            "doc_id": "doc-x",
            "filename": "paper.pdf",
            "source_path": "/tmp/paper.pdf",
            "indexed_at": "2026-01-01T00:00:00+00:00",
            "total_pages": 5,
        }

        client = MagicMock()
        model = MagicMock()
        model.encode.return_value = [_fake_embedding()]

        stats = {
            "embed_seconds": 0.0,
            "upsert_seconds": 0.0,
            "upsert_batches": 0,
            "chunks_indexed": 0,
        }

        flush_records(
            client=client,
            model=model,
            pending_records=pending,
            upsert_batch_size=512,
            metadata=metadata,
            stats=stats,
            embed_batch_size=32,
        )

        upsert_call = client.upsert.call_args
        self.assertIsNotNone(upsert_call)
        points = upsert_call.kwargs.get("points") or upsert_call[1].get("points") or upsert_call[0][1]
        self.assertEqual(len(points), 1)
        payload = points[0].payload
        self.assertNotIn("text", payload, "Full text must not be stored in Qdrant payload")
        self.assertIn("preview", payload, "Preview must be stored in Qdrant payload")
        self.assertEqual(payload["preview"], full_text[:200])

    def test_chunk_text_written_to_sqlite_before_qdrant_upsert(self):
        """SQLite chunk write must happen before Qdrant upsert."""
        import labvdb
        from indexer import flush_records

        call_order: list[str] = []

        import indexer as indexer_mod

        original_write = indexer_mod.write_chunks
        def tracking_write(rows):
            call_order.append("sqlite")
            original_write(rows)

        client = MagicMock()
        client.upsert.side_effect = lambda **kw: call_order.append("qdrant")

        model = MagicMock()
        model.encode.return_value = [_fake_embedding()]

        stats = {"embed_seconds": 0.0, "upsert_seconds": 0.0, "upsert_batches": 0, "chunks_indexed": 0}
        pending = [(1, 0, "Abstract", "some text")]
        metadata = {
            "doc_id": "doc-y",
            "filename": "x.pdf",
            "source_path": "/tmp/x.pdf",
            "indexed_at": "2026-01-01T00:00:00+00:00",
            "total_pages": 1,
        }

        with patch.object(indexer_mod, "write_chunks", side_effect=tracking_write):
            flush_records(
                client=client, model=model, pending_records=pending,
                upsert_batch_size=512, metadata=metadata, stats=stats, embed_batch_size=32,
            )

        self.assertEqual(call_order, ["sqlite", "qdrant"],
                         f"Expected sqlite before qdrant, got: {call_order}")

    def test_full_text_retrievable_from_chunk_store_after_flush(self):
        """After flush_records, full text is fetchable from chunk store by chunk_id."""
        import labvdb
        from indexer import flush_records
        from labvdb import make_chunk_id

        full_text = "The mitochondria is the powerhouse of the cell. " * 20
        doc_id = "doc-z"
        pending = [(2, 0, "Results", full_text)]
        metadata = {
            "doc_id": doc_id,
            "filename": "bio.pdf",
            "source_path": "/tmp/bio.pdf",
            "indexed_at": "2026-01-01T00:00:00+00:00",
            "total_pages": 3,
        }

        client = MagicMock()
        model = MagicMock()
        model.encode.return_value = [_fake_embedding()]
        stats = {"embed_seconds": 0.0, "upsert_seconds": 0.0, "upsert_batches": 0, "chunks_indexed": 0}

        flush_records(
            client=client, model=model, pending_records=pending,
            upsert_batch_size=512, metadata=metadata, stats=stats, embed_batch_size=32,
        )

        expected_chunk_id = make_chunk_id(doc_id, 2, 0)
        retrieved = labvdb.fetch_chunk_text(expected_chunk_id)
        self.assertEqual(retrieved, full_text)


class TestDeleteDocumentCleansChunkStore(unittest.TestCase):
    """delete_document() must also remove chunk rows from SQLite."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._db = Path(self._tmp.name) / "chunks.sqlite3"
        import labvdb
        self._original = labvdb.CHUNK_STORE_PATH
        labvdb.CHUNK_STORE_PATH = self._db

    def tearDown(self):
        import labvdb
        labvdb.CHUNK_STORE_PATH = self._original
        self._tmp.cleanup()

    def test_delete_document_removes_chunk_rows(self):
        """delete_document must call delete_chunks_for_doc for the given doc_id."""
        import labvdb
        from indexer import delete_document

        labvdb.write_chunks([
            ("chunk-1", "target-doc", "text one"),
            ("chunk-2", "target-doc", "text two"),
            ("chunk-3", "other-doc", "other text"),
        ])

        client = MagicMock()
        delete_document(client, "target-doc")

        self.assertIsNone(labvdb.fetch_chunk_text("chunk-1"))
        self.assertIsNone(labvdb.fetch_chunk_text("chunk-2"))
        self.assertEqual(labvdb.fetch_chunk_text("chunk-3"), "other text")


class TestSearchApiChunkEndpoint(unittest.TestCase):
    """Search returns preview; /chunk/{chunk_id} returns full text."""

    def _fresh_app(self, mock_client, chunk_store_path):
        for mod in list(sys.modules):
            if "search_api" in mod:
                del sys.modules[mod]
        import labvdb
        labvdb.CHUNK_STORE_PATH = chunk_store_path

        from fastapi.testclient import TestClient
        with (
            patch("labvdb.get_client", return_value=mock_client),
            patch("labvdb.ensure_collection"),
        ):
            import search_api
            return TestClient(search_api.app), search_api

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._db = Path(self._tmp.name) / "chunks.sqlite3"
        import labvdb
        self._original = labvdb.CHUNK_STORE_PATH
        labvdb.CHUNK_STORE_PATH = self._db

    def tearDown(self):
        import labvdb
        labvdb.CHUNK_STORE_PATH = self._original
        self._tmp.cleanup()
        for mod in list(sys.modules):
            if "search_api" in mod:
                del sys.modules[mod]

    def test_search_returns_preview_not_text(self):
        """GET /search result items must have 'preview' field, not 'text'."""
        import labvdb

        mock_hit = MagicMock()
        mock_hit.score = 0.9
        mock_hit.payload = {
            "doc_id": "doc-1",
            "filename": "paper.pdf",
            "page": 1,
            "chunk_idx": 0,
            "preview": "short preview text",
        }

        mock_client = MagicMock()
        mock_client.query_points.return_value = MagicMock(points=[mock_hit])

        tc, _ = self._fresh_app(mock_client, self._db)
        with tc as client:
            resp = client.get("/search?q=test")

        self.assertEqual(resp.status_code, 200)
        result = resp.json()["results"][0]
        self.assertIn("preview", result)
        self.assertNotIn("text", result)

    def test_chunk_endpoint_returns_full_text(self):
        """GET /chunk/{chunk_id} returns full text from SQLite."""
        import labvdb
        labvdb.write_chunks([("known-chunk", "doc-1", "this is the full text")])

        mock_client = MagicMock()
        tc, _ = self._fresh_app(mock_client, self._db)
        with tc as client:
            resp = client.get("/chunk/known-chunk")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["text"], "this is the full text")

    def test_chunk_endpoint_returns_404_for_unknown_id(self):
        """GET /chunk/{chunk_id} returns 404 when chunk_id is not in SQLite."""
        import labvdb
        labvdb.ensure_chunk_db()

        mock_client = MagicMock()
        tc, _ = self._fresh_app(mock_client, self._db)
        with tc as client:
            resp = client.get("/chunk/no-such-chunk")

        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
