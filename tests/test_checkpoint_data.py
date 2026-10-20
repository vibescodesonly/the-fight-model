import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import resolve_model_data


class CheckpointDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = Path(self.directory.name) / "data.csv"
        self.data.write_bytes(b"recorded checkpoint data\n")
        self.metadata = {"dataset_sha256": hashlib.sha256(self.data.read_bytes()).hexdigest()}

    def test_missing_original_dataset_uses_identical_current_checkout_data(self):
        pipeline = SimpleNamespace(csv_path=str(self.data.parent / "missing-original.csv"))
        with patch("scripts.research.core.DATA_PATH", str(self.data)):
            resolve_model_data(pipeline, self.metadata)
        self.assertEqual(pipeline.csv_path, str(self.data.resolve()))

    def test_explicit_dataset_overrides_existing_training_path(self):
        old = self.data.parent / "old.csv"
        old.write_bytes(b"different checkout data\n")
        pipeline = SimpleNamespace(csv_path=str(old))
        resolve_model_data(pipeline, self.metadata, self.data)
        self.assertEqual(pipeline.csv_path, str(self.data.resolve()))

    def test_relocation_rejects_different_dataset(self):
        self.data.write_bytes(b"changed since training\n")
        pipeline = SimpleNamespace(csv_path="/absent/checkpoint/data.csv")
        with patch("scripts.research.core.DATA_PATH", str(self.data)):
            with self.assertRaisesRegex(ValueError, "dataset changed"):
                resolve_model_data(pipeline, self.metadata)
        self.assertEqual(pipeline.csv_path, "/absent/checkpoint/data.csv")
