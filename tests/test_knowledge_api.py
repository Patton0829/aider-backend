import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.knowledge import router
from app.services.knowledge import KnowledgeService


class KnowledgeManagementApiTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        root = Path(self.directory.name)
        seed_path = root / "seed.md"
        seed_path.write_text("# 默认知识\n\n## 转辙机\n\n锁闭异常需要检查机械阻力。", encoding="utf-8")
        service = KnowledgeService(seed_path, storage_dir=root / "managed")
        service.reload_if_changed(force=True)

        app = FastAPI()
        app.state.knowledge_service = service
        app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.directory.cleanup()

    def test_upload_info_and_delete_document(self):
        response = self.client.post(
            "/api/knowledge/documents",
            params={"filename": "补充知识.md"},
            content="# 补充知识\n\n## 轴承\n\n轴承异响时检查振动频谱。".encode(),
            headers={"Content-Type": "application/octet-stream"},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["documentCount"], 2)

        info = self.client.get("/api/knowledge/info").json()
        self.assertEqual(info["documentCount"], 2)
        self.assertEqual(len(info["documents"]), 2)

        response = self.client.delete("/api/knowledge/documents/补充知识.md")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documentCount"], 1)

    def test_rejects_duplicate_document(self):
        response = self.client.post(
            "/api/knowledge/documents",
            params={"filename": "seed.md"},
            content=b"# duplicate",
            headers={"Content-Type": "application/octet-stream"},
        )

        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
