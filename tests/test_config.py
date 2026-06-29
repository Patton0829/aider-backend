import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.config import Settings


class SettingsTest(unittest.TestCase):
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
