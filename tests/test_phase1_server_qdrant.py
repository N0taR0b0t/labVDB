"""
Phase 1 tests: server Qdrant configuration.

All tests must fail before implementation. Run with:
    python -m pytest tests/test_phase1_server_qdrant.py -v
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, call, patch

import httpx


class TestGetClient(unittest.TestCase):
    """get_client() behaviour depending on env vars."""

    def test_uses_server_url_when_qdrant_url_is_set(self):
        """When QDRANT_URL is set, client connects to that URL."""
        import labvdb
        with patch.dict(os.environ, {"QDRANT_URL": "http://localhost:6333"}, clear=False):
            with patch("labvdb.QdrantClient") as MockClient:
                labvdb.get_client()
                MockClient.assert_called_once_with(url="http://localhost:6333")

    def test_uses_embedded_path_when_qdrant_url_is_absent(self):
        """When QDRANT_URL is not set, client uses local path."""
        import labvdb
        env = {k: v for k, v in os.environ.items() if k != "QDRANT_URL"}
        with patch.dict(os.environ, env, clear=True):
            with patch("labvdb.QdrantClient") as MockClient:
                labvdb.get_client()
                MockClient.assert_called_once_with(path=labvdb.QDRANT_PATH)

    def test_does_not_fall_back_to_embedded_when_url_is_set(self):
        """If QDRANT_URL is set and server is unreachable, must NOT silently use embedded path."""
        import labvdb
        with patch.dict(os.environ, {"QDRANT_URL": "http://localhost:6333"}, clear=False):
            with patch("labvdb.QdrantClient", side_effect=ConnectionError("refused")):
                with self.assertRaises(ConnectionError):
                    labvdb.get_client()


class TestEnsurePayloadIndexes(unittest.TestCase):
    """ensure_payload_indexes() exception handling."""

    def test_already_exists_exception_is_silenced(self):
        """UnexpectedResponse with 400 status (index already exists) must not propagate."""
        from qdrant_client.http.exceptions import UnexpectedResponse
        import labvdb

        client = MagicMock()
        already_exists = UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=b'{"status":{"error":"Index already exists"}}',
            headers=httpx.Headers({}),
        )
        client.create_payload_index.side_effect = already_exists

        # Should not raise
        labvdb.ensure_payload_indexes(client)

    def test_unexpected_server_error_propagates(self):
        """A real server error (5xx) must not be swallowed."""
        from qdrant_client.http.exceptions import UnexpectedResponse
        import labvdb

        client = MagicMock()
        server_error = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b'{"status":{"error":"disk full"}}',
            headers=httpx.Headers({}),
        )
        client.create_payload_index.side_effect = server_error

        with self.assertRaises(UnexpectedResponse):
            labvdb.ensure_payload_indexes(client)

    def test_non_qdrant_exception_propagates(self):
        """Any non-Qdrant exception (e.g. network error) must not be swallowed."""
        import labvdb

        client = MagicMock()
        client.create_payload_index.side_effect = OSError("connection reset")

        with self.assertRaises(OSError):
            labvdb.ensure_payload_indexes(client)


class TestEnsureCollection(unittest.TestCase):
    """ensure_collection() creates collection with quantization on first run."""

    def test_new_collection_created_with_scalar_quantization(self):
        """First-time collection creation must include scalar quantization config."""
        from qdrant_client.http.exceptions import UnexpectedResponse
        from qdrant_client.models import ScalarQuantization
        import labvdb

        client = MagicMock()
        # Simulate collection not existing
        client.get_collection.side_effect = Exception("not found")

        with patch.object(labvdb, "ensure_payload_indexes"):
            labvdb.ensure_collection(client)

        create_call = client.create_collection.call_args
        self.assertIsNotNone(create_call, "create_collection was not called")

        kwargs = create_call.kwargs if create_call.kwargs else create_call[1]
        # on_disk_payload must be True
        self.assertTrue(
            kwargs.get("on_disk_payload"),
            "on_disk_payload must be True when creating collection",
        )
        # quantization_config must include scalar quantization
        q_config = kwargs.get("quantization_config")
        self.assertIsNotNone(q_config, "quantization_config must be provided")
        self.assertIsInstance(
            q_config,
            ScalarQuantization,
            f"Expected ScalarQuantization, got {type(q_config)}",
        )

    def test_existing_collection_is_not_recreated(self):
        """If the collection already exists, create_collection must not be called."""
        import labvdb

        client = MagicMock()
        client.get_collection.return_value = MagicMock()  # exists

        with patch.object(labvdb, "ensure_payload_indexes"):
            labvdb.ensure_collection(client)

        client.create_collection.assert_not_called()

    def test_existing_collection_without_quantization_logs_warning(self):
        """Existing collection lacking quantization config must emit a warning."""
        import logging
        import labvdb

        # Collection exists but has no quantization config
        mock_collection = MagicMock()
        mock_collection.config.quantization_config = None
        client = MagicMock()
        client.get_collection.return_value = mock_collection

        with patch.object(labvdb, "ensure_payload_indexes"):
            with self.assertLogs("labvdb", level=logging.WARNING) as log_ctx:
                labvdb.ensure_collection(client)

        self.assertTrue(
            any("quantization" in msg.lower() for msg in log_ctx.output),
            f"Expected quantization warning, got: {log_ctx.output}",
        )


class TestSearchApiLifespan(unittest.TestCase):
    """search_api client must live inside a lifespan handler, not at import time."""

    def test_client_not_created_at_module_import_time(self):
        """Importing search_api must not call get_client() or ensure_collection()."""
        import sys
        # Remove cached module so we get a fresh import
        for mod_name in list(sys.modules.keys()):
            if "search_api" in mod_name:
                del sys.modules[mod_name]

        with patch("labvdb.get_client") as mock_get_client, \
             patch("labvdb.ensure_collection") as mock_ensure:
            import search_api  # noqa: F401
            mock_get_client.assert_not_called()
            mock_ensure.assert_not_called()

    def test_health_endpoint_exists(self):
        """GET /health must return 200 with status ok when Qdrant is reachable."""
        import sys
        for mod_name in list(sys.modules.keys()):
            if "search_api" in mod_name:
                del sys.modules[mod_name]

        from fastapi.testclient import TestClient

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()

        with patch("labvdb.get_client", return_value=mock_client), \
             patch("labvdb.ensure_collection"):
            import search_api
            with TestClient(search_api.app) as tc:
                response = tc.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
