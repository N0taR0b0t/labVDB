"""
Phase 2 tests: manifest as authority, no whole-collection Qdrant scans.

Run with:
    python -m pytest tests/test_phase2_manifest_authority.py -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch


def _make_manifest_entry(
    path: str,
    *,
    size: int,
    mtime_ns: int,
    doc_id: str | None,
    status: str = "indexed",
    indexed_at: str | None = "2026-01-01T00:00:00+00:00",
    last_error: str | None = None,
):
    """Return a ManifestEntry-like object."""
    from labvdb import ManifestEntry
    return ManifestEntry(
        path=path,
        size=size,
        mtime_ns=mtime_ns,
        doc_id=doc_id,
        indexed_at=indexed_at,
        status=status,
        last_error=last_error,
    )


class TestPlanIndexJobsNoScroll(unittest.TestCase):
    """plan_index_jobs() must never call fetch_doc_ids()."""

    def test_already_indexed_file_is_skipped_without_qdrant_scroll(self):
        """File with matching size/mtime and status=indexed must be skipped; no Qdrant scroll."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "paper.pdf"
            pdf_path.write_bytes(b"content")
            stat = pdf_path.stat()

            manifest = {
                str(pdf_path.resolve()): _make_manifest_entry(
                    str(pdf_path.resolve()),
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    doc_id="known-doc",
                    status="indexed",
                )
            }

            with (
                patch("indexer.load_manifest_entries", return_value=manifest),
                patch("indexer.fetch_doc_ids") as mock_scroll,
                patch("indexer.update_manifest_entry"),
            ):
                from indexer import plan_index_jobs
                jobs, _ = plan_index_jobs(
                    client=object(),
                    pdf_paths=[pdf_path],
                    force=False,
                )

            self.assertEqual(jobs, [], "Already-indexed file must not be queued")
            mock_scroll.assert_not_called()

    def test_new_file_queued_without_qdrant_scroll(self):
        """File with no manifest entry must be queued; no Qdrant scroll."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "new.pdf"
            pdf_path.write_bytes(b"new content")

            with (
                patch("indexer.load_manifest_entries", return_value={}),
                patch("indexer.fetch_doc_ids") as mock_scroll,
                patch("indexer.compute_doc_id", return_value="fresh-doc"),
                patch("indexer.update_manifest_entry"),
            ):
                from indexer import plan_index_jobs
                jobs, _ = plan_index_jobs(
                    client=object(),
                    pdf_paths=[pdf_path],
                    force=False,
                )

            self.assertEqual([j.doc_id for j in jobs], ["fresh-doc"])
            mock_scroll.assert_not_called()

    def test_changed_file_deletes_old_doc_id_without_scroll(self):
        """Changed file must delete old doc_id unconditionally; no Qdrant scroll."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "paper.pdf"
            pdf_path.write_bytes(b"updated content")
            stat = pdf_path.stat()

            manifest = {
                str(pdf_path.resolve()): _make_manifest_entry(
                    str(pdf_path.resolve()),
                    size=stat.st_size - 1,   # size mismatch → stale
                    mtime_ns=stat.st_mtime_ns - 1,
                    doc_id="old-doc",
                    status="indexed",
                )
            }

            deleted: list[str] = []

            with (
                patch("indexer.load_manifest_entries", return_value=manifest),
                patch("indexer.fetch_doc_ids") as mock_scroll,
                patch("indexer.compute_doc_id", return_value="new-doc"),
                patch("indexer.update_manifest_entry"),
                patch("indexer.delete_document", side_effect=lambda c, d: deleted.append(d)),
            ):
                from indexer import plan_index_jobs
                jobs, planning = plan_index_jobs(
                    client=object(),
                    pdf_paths=[pdf_path],
                    force=False,
                )

            self.assertEqual(deleted, ["old-doc"])
            self.assertEqual([j.doc_id for j in jobs], ["new-doc"])
            mock_scroll.assert_not_called()

    def test_force_reindex_deletes_then_queues_without_scroll(self):
        """--force must delete existing doc_id and re-queue; no Qdrant scroll."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "paper.pdf"
            pdf_path.write_bytes(b"content")
            stat = pdf_path.stat()

            manifest = {
                str(pdf_path.resolve()): _make_manifest_entry(
                    str(pdf_path.resolve()),
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    doc_id="existing-doc",
                    status="indexed",
                )
            }

            deleted: list[str] = []

            with (
                patch("indexer.load_manifest_entries", return_value=manifest),
                patch("indexer.fetch_doc_ids") as mock_scroll,
                patch("indexer.update_manifest_entry"),
                patch("indexer.delete_document", side_effect=lambda c, d: deleted.append(d)),
            ):
                from indexer import plan_index_jobs
                jobs, _ = plan_index_jobs(
                    client=object(),
                    pdf_paths=[pdf_path],
                    force=True,
                )

            self.assertEqual(deleted, ["existing-doc"])
            self.assertEqual([j.doc_id for j in jobs], ["existing-doc"])
            mock_scroll.assert_not_called()


class TestStatsUsesManifest(unittest.TestCase):
    """GET /stats document_count must come from manifest, not a Qdrant scroll."""

    def test_stats_document_count_from_manifest_not_scroll(self):
        import sys
        for mod in list(sys.modules):
            if "search_api" in mod:
                del sys.modules[mod]

        from fastapi.testclient import TestClient

        mock_client = MagicMock()
        mock_client.count.return_value = MagicMock(count=4929)

        with (
            patch("labvdb.get_client", return_value=mock_client),
            patch("labvdb.ensure_collection"),
            patch("labvdb.count_unique_doc_ids") as mock_scroll,
            patch("labvdb.count_indexed_docs", return_value=178),
        ):
            import search_api
            with TestClient(search_api.app) as tc:
                response = tc.get("/stats")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_count"], 178)
        mock_scroll.assert_not_called()


class TestCountIndexedDocs(unittest.TestCase):
    """count_indexed_docs() reads from manifest SQLite, not Qdrant."""

    def test_counts_only_indexed_status_rows(self):
        import sqlite3, tempfile
        from pathlib import Path
        import labvdb

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "CREATE TABLE manifest "
                    "(path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, "
                    "doc_id TEXT, indexed_at TEXT, status TEXT NOT NULL, last_error TEXT)"
                )
                conn.executemany(
                    "INSERT INTO manifest VALUES (?,?,?,?,?,?,?)",
                    [
                        ("/a.pdf", 1, 1, "doc-a", "2026-01-01", "indexed", None),
                        ("/b.pdf", 1, 1, "doc-b", "2026-01-01", "indexed", None),
                        ("/c.pdf", 1, 1, None, None, "failed", "parse error"),
                        ("/d.pdf", 1, 1, "doc-d", "2026-01-01", "duplicate_local", None),
                    ],
                )

            original = labvdb.MANIFEST_PATH
            labvdb.MANIFEST_PATH = db_path
            try:
                result = labvdb.count_indexed_docs()
            finally:
                labvdb.MANIFEST_PATH = original

        self.assertEqual(result, 2)
