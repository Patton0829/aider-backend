import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.config import PROJECT_ROOT, Settings


class SettingsTest(unittest.TestCase):
    def test_data_paths_are_relative_to_checkout_not_macos_or_working_directory(self):
        settings = Settings(
            knowledge_milvus_uri=Path("storage/rag/milvus.db"),
            knowledge_index_catalog_path=Path("storage/rag/catalog.sqlite3"),
            knowledge_base_dir=Path("knowledge-bases"),
            workspace_root=Path(".."),
        )
        self.assertEqual(settings.knowledge_milvus_uri, PROJECT_ROOT / "storage/rag/milvus.db")
        self.assertEqual(settings.knowledge_index_catalog_path, PROJECT_ROOT / "storage/rag/catalog.sqlite3")
        self.assertEqual(settings.knowledge_base_dir, PROJECT_ROOT / "knowledge-bases")
        self.assertEqual(settings.workspace_root, PROJECT_ROOT.parent)

    def test_workspace_allowlist(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "transfer"
            workspace.mkdir()
            settings = Settings(
                workspace_root=root,
                allowed_workspaces="transfer",
            )

            self.assertEqual(settings.resolve_workspace("transfer"), workspace.resolve())

            with self.assertRaises(ValueError):
                settings.resolve_workspace("control-center-be")
