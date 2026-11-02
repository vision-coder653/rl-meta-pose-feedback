"""Verify download integrity handling and publication-local resource lookup."""
from pathlib import Path
import importlib.util
import io
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('download_meta', ROOT / 'tools/download_meta.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)
sys.path.insert(0, str(ROOT / 'core'))
from emg_control_paths import resolve_experiment_path


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = ROOT / '.cache/preparation_tests'
        cls.cache.mkdir(parents=True, exist_ok=True)

    def make_archive(self, folder, payload=b'synthetic checkpoint'):
        path = Path(folder) / 'model.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            entry = tarfile.TarInfo('emg2pose_model_checkpoints/tracking_vemg2pose.ckpt')
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
            extra = tarfile.TarInfo('../../unexpected.txt')
            extra.size = 6
            archive.addfile(extra, io.BytesIO(b'unused'))
        return path

    def test_verified_model_is_only_extracted_member(self):
        import hashlib
        with tempfile.TemporaryDirectory(dir=self.cache) as folder:
            archive = self.make_archive(folder)
            output = Path(folder) / 'weights' / download.MODEL_NAME
            with patch.object(download, 'ARCHIVE_SHA256', download.sha256(archive)), \
                 patch.object(download, 'MODEL_SHA256', hashlib.sha256(b'synthetic checkpoint').hexdigest()):
                download.prepare(archive, output)
                self.assertEqual(output.read_bytes(), b'synthetic checkpoint')
                download.prepare(archive, output)
            self.assertEqual({p.name for p in output.parent.iterdir()}, {download.MODEL_NAME})

    def test_rejects_bad_archive_before_extracting(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as folder:
            archive = self.make_archive(folder)
            output = Path(folder) / 'weights' / download.MODEL_NAME
            with self.assertRaisesRegex(ValueError, 'archive checksum'):
                download.prepare(archive, output)
            self.assertFalse(output.exists())

    def test_rejects_bad_model_and_cleans_own_temporary(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as folder:
            archive = self.make_archive(folder)
            output = Path(folder) / 'weights' / download.MODEL_NAME
            with patch.object(download, 'ARCHIVE_SHA256', download.sha256(archive)):
                with self.assertRaisesRegex(ValueError, 'model checksum'):
                    download.prepare(archive, output)
            self.assertEqual(list(output.parent.iterdir()), [])

    def test_resources_resolve_within_publication(self):
        for path in ('resources/robot/hand_f1_990_f2_990_f3_990_f4_990_f5_990_complete.xml',
                     'configs/training_groups.json', 'experiments/rl_meta_pose_feedback/configs/encoder.json'):
            resolved = resolve_experiment_path(path)
            self.assertTrue(resolved.is_file(), path)
            self.assertTrue(resolved.is_relative_to(ROOT))


if __name__ == '__main__':
    unittest.main()
