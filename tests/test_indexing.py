from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from indexer import plan_index_jobs, resolve_pdf_paths
from labvdb import CHUNK_OVERLAP_BLOCKS, chunk_page_text


class FakePage:
    def __init__(self, blocks: list[str]):
        self._blocks = blocks

    def get_text(self, mode: str):
        if mode != "blocks":
            raise ValueError(mode)
        rows = []
        for idx, text in enumerate(self._blocks):
            rows.append((0, float(idx), 100, float(idx) + 1.0, text, 0, 0))
        return rows


class ChunkingTests(unittest.TestCase):
    def test_chunk_overlap_carries_previous_block_forward(self) -> None:
        long_a = "A" * 900
        long_b = "B" * 900
        long_c = "C" * 900
        page = FakePage([long_a, long_b, long_c])

        chunks = chunk_page_text(page, page_num=1)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, f"{long_a}\n\n{long_b}")
        self.assertEqual(chunks[1].text, f"{long_a}\n\n{long_b}")
        self.assertEqual(chunks[2].text, f"{long_b}\n\n{long_c}")
        self.assertEqual(CHUNK_OVERLAP_BLOCKS, 1)


class IndexPlanningTests(unittest.TestCase):
    def test_changed_file_replaces_previous_doc_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "paper.pdf"
            pdf_path.write_bytes(b"new content")
            stat = pdf_path.stat()
            manifest_entry = {
                str(pdf_path.resolve()): type(
                    "ManifestEntryStub",
                    (),
                    {
                        "size": stat.st_size - 1,
                        "mtime_ns": stat.st_mtime_ns - 1,
                        "doc_id": "old-doc",
                    },
                )()
            }

            deleted_doc_ids: list[str] = []

            def fake_delete_document(client, doc_id: str) -> None:
                deleted_doc_ids.append(doc_id)

            with (
                patch("indexer.load_manifest_entries", return_value=manifest_entry),
                patch("indexer.fetch_doc_ids", return_value={"old-doc"}),
                patch("indexer.compute_doc_id", return_value="new-doc"),
                patch("indexer.update_manifest_entry"),
                patch("indexer.delete_document", side_effect=fake_delete_document),
            ):
                jobs, planning = plan_index_jobs(
                    client=object(),
                    pdf_paths=[pdf_path],
                    force=False,
                )

            self.assertEqual([job.doc_id for job in jobs], ["new-doc"])
            self.assertEqual(deleted_doc_ids, ["old-doc"])
            self.assertEqual(
                planning["replaced_doc_ids"],
                [(pdf_path, "old-doc", "new-doc")],
            )

    def test_resolve_pdf_paths_accepts_case_insensitive_pdf_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "paper.PDF"
            txt_path = Path(tmp_dir) / "notes.txt"
            pdf_path.write_text("x")
            txt_path.write_text("x")

            args = Namespace(pdf_dir=Path(tmp_dir), file=[pdf_path, txt_path])
            resolved = resolve_pdf_paths(args)

            self.assertEqual(resolved, [pdf_path])


if __name__ == "__main__":
    unittest.main()
