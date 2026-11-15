import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/configure_mysql_history.py"


class HistoryProvisioningTest(unittest.TestCase):
    def run_script(self, root, failure=False):
        spec = importlib.util.spec_from_file_location("history_provision", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = io.StringIO()
        side_effect = CalledProcessError(1, "mysql") if failure else None
        with patch.object(module, "__file__", str(root / "scripts/configure_mysql_history.py")), \
             patch("sys.argv", ["configure_mysql_history.py"]), \
             patch.object(module.secrets, "token_hex", return_value="f" * 48), \
             patch.object(module.subprocess, "run", side_effect=side_effect) as command, redirect_stdout(output):
            result = module.main()
        return result, output.getvalue(), command

    def test_password_uses_stdin_and_private_environment_file(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".env").write_text("DASHSCOPE_API_KEY=original-test-value\nKNOWLEDGE_HISTORY_STORE=off\n")
            result, output, command = self.run_script(root)
            self.assertEqual(result, 0)
            self.assertNotIn("f" * 48, str(command.call_args.args))
            self.assertIn("f" * 48, command.call_args.kwargs["input"])
            self.assertNotIn("f" * 48, output)
            contents = (root / ".env").read_text()
            self.assertIn("DASHSCOPE_API_KEY=original-test-value", contents)
            self.assertIn("KNOWLEDGE_HISTORY_STORE=mysql", contents)
            self.assertEqual((root / ".env").stat().st_mode & 0o777, 0o600)

    def test_mysql_failure_preserves_original_environment(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            original = "DASHSCOPE_API_KEY=original-test-value\nKNOWLEDGE_HISTORY_STORE=off\n"
            (root / ".env").write_text(original)
            result, output, _ = self.run_script(root, failure=True)
            self.assertEqual(result, 1)
            self.assertEqual((root / ".env").read_text(), original)
            self.assertNotIn("f" * 48, output)
            self.assertEqual(list((root / "storage/rag").glob(".history-env-*")), [])
