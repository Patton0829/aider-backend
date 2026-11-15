import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError
from sqlalchemy.engine import make_url

from app.core.config import PROJECT_ROOT, Settings


class SettingsTest(unittest.TestCase):
    def test_database_parts_encode_password_and_hide_secrets(self):
        password = 'p@ss:/?#% +密码"end'
        settings = Settings(db_host="db.example.com", db_username="rag_user",
                            db_password=password, knowledge_history_store="mysql")
        url = make_url(settings.knowledge_database_url)
        self.assertEqual(url.drivername, "mysql+asyncmy")
        self.assertEqual((url.host, url.port, url.database, url.username),
                         ("db.example.com", 3306, "aider_rag", "rag_user"))
        self.assertEqual(url.password, password)
        self.assertNotIn(password, repr(settings))
        self.assertNotIn(settings.knowledge_database_url, repr(settings))

    def test_explicit_database_url_keeps_precedence_over_parts(self):
        original = "mysql+asyncmy://existing:password@old.example.com:3308/existing_db"
        settings = Settings(knowledge_database_url=original, db_host="new.example.com",
                            db_username="new_user", db_password="different")
        self.assertEqual(settings.knowledge_database_url, original)

    def test_empty_database_config_does_not_enable_a_connection(self):
        settings = Settings()
        self.assertEqual(settings.knowledge_database_url, "")
        self.assertEqual(settings.knowledge_history_store, "off")

    def test_database_parts_validate_required_fields_and_port(self):
        for values in (
            {"db_host": "db.example.com"},
            {"db_host": "db.example.com", "db_username": "user", "db_database": " "},
            {"db_port": 0}, {"db_port": 65536},
        ):
            with self.subTest(fields=tuple(values)):
                with self.assertRaises(ValidationError):
                    Settings(**values)

    def test_database_parts_from_env_with_spaces_and_environment_override(self):
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                'KNOWLEDGE_HISTORY_STORE = mysql\n'
                'KNOWLEDGE_DATABASE_URL = \n'
                'DB_HOST = file.example.com\nDB_PORT = 3308\n'
                'DB_DATABASE = aider_rag\nDB_USERNAME = rag_user\n'
                'DB_PASSWORD = "p@ss #:/% word"\n', encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                url = make_url(Settings.from_env(env_file).knowledge_database_url)
                self.assertEqual(url.password, "p@ss #:/% word")
                self.assertEqual(url.host, "file.example.com")
                self.assertEqual(url.port, 3308)
            with patch.dict("os.environ", {"DB_HOST": "process.example.com",
                                          "DB_PASSWORD": 'different@:/#"'}, clear=True):
                url = make_url(Settings.from_env(env_file).knowledge_database_url)
                self.assertEqual(url.host, "process.example.com")
                self.assertEqual(url.password, 'different@:/#"')

    def test_env_password_preserves_literal_quotes_and_quoted_spaces(self):
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            for raw, expected in (("ends-with-quote'", "ends-with-quote'"),
                                  ('"  spaces #@  "', "  spaces #@  "),
                                  ('\'"both-quotes"\'', '"both-quotes"')):
                with self.subTest(raw=raw):
                    env_file.write_text("DB_HOST=db.example.com\nDB_USERNAME=user\nDB_PASSWORD=" + raw + "\n")
                    with patch.dict("os.environ", {}, clear=True):
                        url = make_url(Settings.from_env(env_file).knowledge_database_url)
                        self.assertEqual(url.password, expected)

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
