import unittest
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.config import Settings
from app.services.job_logs import JobLogService


class JobLogServiceTest(unittest.TestCase):
    def test_prefers_live_log(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "12" / "output"
            output.mkdir(parents=True)
            (output / "other.out").write_text("other", encoding="utf-8")
            live_log = output / "live.log"
            live_log.write_text("live", encoding="utf-8")

            service = JobLogService(Settings(job_workspace_root=root))

            self.assertEqual(service.find_log_file(12), live_log.resolve())


class JobLogFollowTest(unittest.IsolatedAsyncioTestCase):
    async def test_emits_snapshot_then_appended_content(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "7" / "output"
            output.mkdir(parents=True)
            live_log = output / "live.log"
            live_log.write_text("epoch 1\n", encoding="utf-8")

            service = JobLogService(
                Settings(job_workspace_root=root, job_log_poll_interval=0.01)
            )
            events = service.follow(7)

            self.assertEqual(await anext(events), ("snapshot", "epoch 1\n"))
            live_log.write_text("epoch 1\nepoch 2\n", encoding="utf-8")
            self.assertEqual(await asyncio.wait_for(anext(events), 1), ("append", "epoch 2\n"))
            await events.aclose()
