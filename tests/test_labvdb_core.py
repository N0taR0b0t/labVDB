"""
Tests for core labvdb.py utilities: doc_id stability, duplicate detection,
is_junk_block, and canonicalize_section.

Run with:
    python -m pytest tests/test_labvdb_core.py -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestDocIdStability(unittest.TestCase):

    def test_same_content_different_filename_same_doc_id(self):
        from labvdb import compute_doc_id
        content = b"identical pdf bytes for testing"
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "version_a.pdf"
            p2 = Path(tmp) / "version_b.pdf"
            p1.write_bytes(content)
            p2.write_bytes(content)
            self.assertEqual(compute_doc_id(p1), compute_doc_id(p2))

    def test_different_content_different_doc_id(self):
        from labvdb import compute_doc_id
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.pdf"
            p2 = Path(tmp) / "b.pdf"
            p1.write_bytes(b"content alpha")
            p2.write_bytes(b"content beta")
            self.assertNotEqual(compute_doc_id(p1), compute_doc_id(p2))

    def test_doc_id_is_40_char_lowercase_hex(self):
        from labvdb import compute_doc_id
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "paper.pdf"
            p.write_bytes(b"some bytes")
            doc_id = compute_doc_id(p)
        self.assertRegex(doc_id, r'^[0-9a-f]{40}$')

    def test_repeated_call_same_file_is_stable(self):
        from labvdb import compute_doc_id
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "paper.pdf"
            p.write_bytes(b"stable content")
            self.assertEqual(compute_doc_id(p), compute_doc_id(p))


class TestDuplicateDetection(unittest.TestCase):

    def test_two_files_same_content_second_marked_duplicate_local(self):
        from indexer import plan_index_jobs
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "copy_a.pdf"
            p2 = Path(tmp) / "copy_b.pdf"
            p1.write_bytes(b"same content")
            p2.write_bytes(b"same content")
            manifest_calls = []
            with (
                patch("indexer.load_manifest_entries", return_value={}),
                patch("indexer.compute_doc_id", return_value="aaaa1111" * 5),
                patch("indexer.update_manifest_entry",
                      side_effect=lambda **kw: manifest_calls.append(kw)),
                patch("indexer.delete_document"),
            ):
                jobs, planning = plan_index_jobs(
                    client=object(), pdf_paths=[p1, p2], force=False
                )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].pdf_path, p1)
        self.assertEqual(len(planning["skipped_duplicate_local"]), 1)
        dup_path, dup_doc_id = planning["skipped_duplicate_local"][0]
        self.assertEqual(dup_path, p2)
        dup_manifest_call = next(
            (c for c in manifest_calls if c.get("status") == "duplicate_local"), None
        )
        self.assertIsNotNone(dup_manifest_call)
        self.assertEqual(dup_manifest_call["doc_id"], "aaaa1111" * 5)

    def test_two_files_different_content_both_queued(self):
        from indexer import plan_index_jobs
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.pdf"
            p2 = Path(tmp) / "b.pdf"
            p1.write_bytes(b"content a")
            p2.write_bytes(b"content b")
            doc_ids = {"a.pdf": "hash-aaa" + "0" * 32, "b.pdf": "hash-bbb" + "0" * 32}
            with (
                patch("indexer.load_manifest_entries", return_value={}),
                patch("indexer.compute_doc_id",
                      side_effect=lambda path: doc_ids[path.name]),
                patch("indexer.update_manifest_entry"),
                patch("indexer.delete_document"),
            ):
                jobs, planning = plan_index_jobs(
                    client=object(), pdf_paths=[p1, p2], force=False
                )

        self.assertEqual(len(jobs), 2)
        self.assertEqual(planning["skipped_duplicate_local"], [])


class TestIsJunkBlock(unittest.TestCase):

    def test_short_text_is_junk(self):
        from labvdb import is_junk_block
        self.assertTrue(is_junk_block("Too short"))

    def test_no_alphabetic_chars_is_junk(self):
        from labvdb import is_junk_block
        self.assertTrue(is_junk_block("1234 5678 9012 3456"))

    def test_page_header_uppercase_plus_page_number_is_junk(self):
        from labvdb import is_junk_block
        self.assertTrue(is_junk_block("FATTY ACID OXIDATION AND KETOGENESIS\n403"))

    def test_page_header_with_ocr_noise_is_junk(self):
        from labvdb import is_junk_block
        # OCR artefact: lowercase letter in otherwise uppercase header
        self.assertTrue(is_junk_block("FATrY ACID OXIDATION AND KETOGENESIS\n403"))

    def test_too_few_distinct_tokens_is_junk(self):
        from labvdb import is_junk_block
        # Only 2 distinct tokens — below the threshold of 3
        self.assertTrue(is_junk_block("foo foo foo foo foo bar bar bar bar bar"))

    def test_normal_paragraph_is_not_junk(self):
        from labvdb import is_junk_block
        text = (
            "The mitochondria is the powerhouse of the cell and plays a central "
            "role in energy production through oxidative phosphorylation."
        )
        self.assertFalse(is_junk_block(text))

    def test_two_line_second_not_page_number_not_junk(self):
        from labvdb import is_junk_block
        text = (
            "The quick brown fox jumps over the lazy dog today "
            "and tomorrow always finds a way forward"
        )
        self.assertFalse(is_junk_block(text))

    def test_single_word_is_junk(self):
        from labvdb import is_junk_block
        self.assertTrue(is_junk_block("Fig."))
        self.assertTrue(is_junk_block("403"))


class TestCanonicalizeSection(unittest.TestCase):

    def test_exact_alias_lowercase(self):
        from labvdb import canonicalize_section
        self.assertEqual(canonicalize_section("abstract"), "Abstract")
        self.assertEqual(canonicalize_section("summary"), "Abstract")
        self.assertEqual(canonicalize_section("methods"), "Methods")
        self.assertEqual(canonicalize_section("references"), "References")

    def test_case_insensitive_normalisation(self):
        from labvdb import canonicalize_section
        self.assertEqual(canonicalize_section("METHODS"), "Methods")
        self.assertEqual(canonicalize_section("Results"), "Results")
        self.assertEqual(canonicalize_section("Discussion"), "Discussion")

    def test_punctuation_stripped(self):
        from labvdb import canonicalize_section
        self.assertEqual(canonicalize_section("Methods:"), "Methods")
        self.assertEqual(canonicalize_section("3. Results"), "Results")

    def test_numbered_section_heading(self):
        from labvdb import canonicalize_section
        self.assertEqual(canonicalize_section("2.1 Statistical Analysis"), "Methods")
        self.assertEqual(canonicalize_section("1. Introduction"), "Introduction")

    def test_sliding_window_prefix_match(self):
        from labvdb import canonicalize_section
        # "statistical analysis of variance" — direct match fails, but 2-word
        # prefix "statistical analysis" is in SECTION_ALIASES
        self.assertEqual(
            canonicalize_section("statistical analysis of variance"), "Methods"
        )

    def test_multi_word_alias(self):
        from labvdb import canonicalize_section
        self.assertEqual(canonicalize_section("Materials and Methods"), "Methods")
        self.assertEqual(canonicalize_section("Patients and Methods"), "Methods")
        self.assertEqual(canonicalize_section("Results and Discussion"), "Results")
        self.assertEqual(canonicalize_section("Acknowledgements"), "Acknowledgments")
        self.assertEqual(canonicalize_section("Supporting Information"), "Supplementary")

    def test_unknown_heading_returns_none(self):
        from labvdb import canonicalize_section
        self.assertIsNone(canonicalize_section("Author Contributions"))
        self.assertIsNone(canonicalize_section("Graphical Abstract Overview"))

    def test_empty_or_whitespace_returns_none(self):
        from labvdb import canonicalize_section
        self.assertIsNone(canonicalize_section(""))
        self.assertIsNone(canonicalize_section("   "))


class TestFtsLexicalScores(unittest.TestCase):
    """Tests for the FTS5-backed BM25 lexical scorer."""

    def setUp(self):
        import labvdb
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_path = labvdb.CHUNK_STORE_PATH
        labvdb.CHUNK_STORE_PATH = Path(self._tmp.name) / "chunks.sqlite3"

    def tearDown(self):
        import labvdb
        labvdb.CHUNK_STORE_PATH = self._orig_path
        self._tmp.cleanup()

    def _write(self, rows: list[tuple[str, str, str]]) -> None:
        from labvdb import write_chunks
        write_chunks(rows)

    def test_matching_chunk_gets_nonzero_score(self):
        from labvdb import fts_lexical_scores
        self._write([("id1", "doc1", "Fatty acid oxidation occurs in the mitochondria.")])
        scores = fts_lexical_scores("fatty acid oxidation", ["id1"])
        self.assertGreater(scores.get("id1", 0.0), 0.0)

    def test_no_match_returns_zero(self):
        from labvdb import fts_lexical_scores
        self._write([("id1", "doc1", "The mitochondria processes fatty acids.")])
        scores = fts_lexical_scores("quantum mechanics", ["id1"])
        self.assertEqual(scores.get("id1", 0.0), 0.0)

    def test_better_match_scores_higher(self):
        from labvdb import fts_lexical_scores
        self._write([
            ("id1", "doc1", "Sphingomyelinase cleaves sphingomyelin into ceramide and phosphocholine."),
            ("id2", "doc2", "The patient received standard care and monitoring."),
        ])
        scores = fts_lexical_scores("sphingomyelinase sphingomyelin ceramide", ["id1", "id2"])
        self.assertGreater(scores.get("id1", 0.0), scores.get("id2", 0.0))

    def test_best_match_normalised_to_one(self):
        from labvdb import fts_lexical_scores
        self._write([
            ("id1", "doc1", "Acyl-CoA synthetase catalyzes the formation of acyl-CoA thioesters."),
            ("id2", "doc2", "Carbohydrate metabolism follows different enzymatic routes."),
        ])
        scores = fts_lexical_scores("acyl-CoA metabolism", ["id1", "id2"])
        self.assertGreater(len(scores), 0)
        self.assertAlmostEqual(max(scores.values()), 1.0)

    def test_chunk_id_not_in_fts_gets_zero(self):
        from labvdb import fts_lexical_scores
        self._write([("id1", "doc1", "Fatty acid beta oxidation pathway in peroxisomes.")])
        scores = fts_lexical_scores("fatty acid", ["id1", "ghost-id"])
        self.assertEqual(scores.get("ghost-id", 0.0), 0.0)

    def test_bm25_not_fooled_by_term_repetition(self):
        """BM25 length normalisation should not over-reward pathological repetition."""
        from labvdb import fts_lexical_scores
        chunk_informative = (
            "Sphingomyelinase activity was significantly elevated in lipopolysaccharide-treated cells, "
            "suggesting a role in ceramide-mediated apoptosis signalling."
        )
        # Repeat the query term 20× with no other content — BM25 penalises this via length norm.
        chunk_spam = " ".join(["sphingomyelinase"] * 20)
        self._write([
            ("id_info", "doc1", chunk_informative),
            ("id_spam", "doc2", chunk_spam),
        ])
        scores = fts_lexical_scores("sphingomyelinase activity", ["id_info", "id_spam"])
        # Informative chunk should score at least as well as the spam chunk.
        self.assertGreaterEqual(scores.get("id_info", 0.0), scores.get("id_spam", 0.0))

    def test_fts_backfilled_from_pre_existing_chunks(self):
        """Chunks written before the FTS5 table existed must be reachable after ensure_chunk_db."""
        import sqlite3
        import labvdb
        db_path = labvdb.CHUNK_STORE_PATH
        # Write chunk directly to the raw table, bypassing FTS5 triggers.
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS chunks "
                "(chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, full_text TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?)",
                ("pre-id", "doc0", "20-HETE is a potent vasoconstrictor in renal arterioles."),
            )
            conn.commit()
        # Calling ensure_chunk_db should create FTS5 and backfill from existing rows.
        labvdb.ensure_chunk_db()
        scores = labvdb.fts_lexical_scores("20-HETE vasoconstrictor", ["pre-id"])
        self.assertGreater(scores.get("pre-id", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
