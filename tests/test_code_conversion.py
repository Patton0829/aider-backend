import asyncio
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from app.code_conversion.engine import CodeConversionEngine, PackageValidationError
from app.code_conversion.service import CodeConversionService, ConversionUploadError
from app.core.config import Settings


class FakeCompletions:
    def __init__(self, files=None):
        self.files = files or {}
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps(
            {"files": self.files, "notes": "converted in test"},
            ensure_ascii=False,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self, files=None):
        self.chat = SimpleNamespace(completions=FakeCompletions(files))


def make_zip(path: Path, files: dict[str, str | bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


class CodeConversionEngineTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    async def asyncTearDown(self):
        self.temp.cleanup()

    def settings(self) -> Settings:
        return Settings(
            code_conversion_storage_dir=self.root / "storage",
            code_conversion_max_source_bytes=1024 * 1024,
        )

    async def test_python_entry_skips_model_conversion(self):
        archive = self.root / "python.zip"
        make_zip(
            archive,
            {
                "algorithm/algorithm.json": json.dumps(
                    {
                        "name": "demo",
                        "displayName": "Demo",
                        "version": "1.0.0",
                        "type": "python",
                        "entry": "main.py",
                    }
                ),
                "algorithm/main.py": "print('ok')\n",
            },
        )
        client = FakeClient()
        engine = CodeConversionEngine(self.settings(), client)
        events = []

        async def progress(percent, phase, message, details=None):
            events.append((percent, phase, message, details))

        outcome = await engine.convert(archive, self.root / "job-python", progress)
        self.assertFalse(outcome.conversion_required)
        self.assertEqual(outcome.source_language, "python")
        self.assertEqual(client.chat.completions.calls, [])
        self.assertEqual(events[-1][0], 100)

    async def test_cpp_project_is_converted_and_packaged_as_python(self):
        archive = self.root / "cpp.zip"
        make_zip(
            archive,
            {
                "algorithm/algorithm.json": json.dumps(
                    {
                        "name": "demo_cpp",
                        "displayName": "Demo C++",
                        "version": "1.0.0",
                        "type": "cpp",
                        "entry": "main.cpp",
                    }
                ),
                "algorithm/main.cpp": "int main() { return 0; }\n",
                "algorithm/helper.hpp": "int answer();\n",
                "algorithm/data.txt": "keep me\n",
            },
        )
        client = FakeClient(
            {
                "main.py": "def main():\n    return 0\n\nif __name__ == '__main__':\n    main()\n",
                "helper.py": "def answer():\n    return 42\n",
            }
        )
        engine = CodeConversionEngine(self.settings(), client)
        events = []

        async def progress(percent, phase, message, details=None):
            events.append((percent, phase, message, details))

        job_dir = self.root / "job-cpp"
        outcome = await engine.convert(archive, job_dir, progress)
        self.assertTrue(outcome.conversion_required)
        self.assertTrue(outcome.artifact_available)
        self.assertEqual(outcome.source_language, "cpp")
        with zipfile.ZipFile(job_dir / "artifact.zip") as converted:
            names = set(converted.namelist())
            self.assertIn("algorithm/main.py", names)
            self.assertIn("algorithm/helper.py", names)
            self.assertIn("algorithm/data.txt", names)
            self.assertNotIn("algorithm/main.cpp", names)
            self.assertNotIn("algorithm/helper.hpp", names)
            manifest = json.loads(converted.read("algorithm/algorithm.json"))
        self.assertEqual(manifest["type"], "python")
        self.assertEqual(manifest["entry"], "main.py")
        self.assertEqual(manifest["conversion"]["sourceLanguage"], "cpp")
        self.assertTrue(any("helper.hpp" in event[2] for event in events))
        call = client.chat.completions.calls[0]
        self.assertFalse(call["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    async def test_rejects_parent_path_in_zip(self):
        archive = self.root / "unsafe.zip"
        make_zip(archive, {"../algorithm.json": "{}"})
        engine = CodeConversionEngine(self.settings(), FakeClient())

        async def progress(*_args):
            return None

        with self.assertRaises(PackageValidationError):
            await engine.convert(archive, self.root / "job-unsafe", progress)


class CodeConversionServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_recovers_incomplete_job_from_input_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(code_conversion_storage_dir=root / "storage")
            job_dir = settings.code_conversion_storage_dir / "recover-job"
            job_dir.mkdir(parents=True)
            make_zip(
                job_dir / "input.zip",
                {
                    "algorithm.json": json.dumps(
                        {
                            "name": "demo",
                            "displayName": "Demo",
                            "version": "1.0.0",
                            "type": "python",
                            "entry": "main.py",
                        }
                    ),
                    "main.py": "print('recovered')\n",
                },
            )
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "jobId": "recover-job",
                        "status": "converting",
                        "phase": "converting",
                        "progress": 44,
                        "message": "interrupted",
                        "targetLanguage": "python",
                        "sourceFiles": ["main.py"],
                        "generatedFiles": [],
                        "logs": [],
                        "artifactAvailable": False,
                    }
                ),
                encoding="utf-8",
            )

            service = CodeConversionService(settings, FakeClient())
            self.assertEqual(await service.recover_incomplete_jobs(), 1)
            for _ in range(100):
                record = await service.get("recover-job")
                if record.status in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(record.status, "completed")
            self.assertTrue(any("恢复转换" in entry.message for entry in record.logs))
            await service.close()

    async def test_job_progress_reaches_completed_for_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(code_conversion_storage_dir=root / "storage")
            service = CodeConversionService(settings, FakeClient())
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "algorithm.json",
                    json.dumps(
                        {
                            "name": "demo",
                            "displayName": "Demo",
                            "version": "1.0.0",
                            "type": "python",
                            "entry": "main.py",
                        }
                    ),
                )
                archive.writestr("main.py", "print('ok')\n")

            async def chunks():
                yield buffer.getvalue()

            await service.submit("job-123", "demo.zip", chunks())
            for _ in range(100):
                record = await service.get("job-123")
                if record.status in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(record.status, "completed")
            self.assertEqual(record.progress, 100)
            self.assertFalse(record.conversion_required)
            self.assertGreaterEqual(len(record.logs), 4)
            self.assertTrue(any("main.py" in entry.message for entry in record.logs))
            files = await service.converted_files("job-123")
            self.assertEqual(
                [file.name for file in files],
                ["main.py", "algorithm.json"],
            )
            content = await service.converted_file_content("job-123", "main.py")
            self.assertIn("print('ok')", content.content)
            await service.close()

    async def test_lists_and_reads_only_viewable_converted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(code_conversion_storage_dir=root / "storage")
            service = CodeConversionService(
                settings,
                FakeClient({"main.py": "print('converted')\n"}),
            )
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "demo/algorithm.json",
                    json.dumps(
                        {
                            "name": "demo",
                            "displayName": "Demo",
                            "version": "1.0.0",
                            "type": "cpp",
                            "entry": "main.cpp",
                        }
                    ),
                )
                archive.writestr("demo/main.cpp", "int main() { return 0; }\n")
                archive.writestr("demo/acc.json", '{"0": 0.8}\n')
                archive.writestr("demo/loss.json", '{"0": 0.2}\n')
                archive.writestr("demo/private.bin", b"secret")

            async def chunks():
                yield buffer.getvalue()

            await service.submit("converted-files", "demo.zip", chunks())
            for _ in range(100):
                record = await service.get("converted-files")
                if record.status in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(record.status, "completed")

            files = await service.converted_files("converted-files")
            self.assertEqual(
                [file.name for file in files],
                ["main.py", "algorithm.json", "acc.json", "loss.json"],
            )
            content = await service.converted_file_content(
                "converted-files", "demo/main.py"
            )
            self.assertEqual(content.language, "python")
            self.assertIn("converted", content.content)
            with self.assertRaises(ConversionUploadError):
                await service.converted_file_content(
                    "converted-files", "demo/private.bin"
                )
            with self.assertRaises(ConversionUploadError):
                await service.converted_file_content(
                    "converted-files", "../job.json"
                )
            await service.close()


if __name__ == "__main__":
    unittest.main()
