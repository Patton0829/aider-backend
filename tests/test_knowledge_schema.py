import unittest

from app.schemas.knowledge import KnowledgeEvent, KnowledgeRequest, KnowledgeSource


class KnowledgeSchemaTest(unittest.TestCase):
    def test_persisted_sources_roundtrip_with_camel_case_fields(self):
        source = KnowledgeSource.model_validate({"document": "知识.md", "chapter": "第三章", "section": "3.2", "entry_id": "SW-021", "line_start": 1, "line_end": 2, "score": .9, "kb_version": "a" * 64})
        self.assertEqual(KnowledgeSource.model_validate(source.model_dump(by_alias=True)), source)

    def test_owner_credential_is_excluded_from_request_serialization(self):
        request = KnowledgeRequest.model_validate({"ownerToken": "a" * 64})
        self.assertNotIn("a" * 64, str(request.model_dump()))
    def test_request_accepts_camel_case_request_id(self):
        request = KnowledgeRequest.model_validate(
            {
                "type": "ask",
                "question": "牵引电机温度异常如何排查？",
                "requestId": "diagnosis-2",
            }
        )

        self.assertEqual(request.request_id, "diagnosis-2")

    def test_event_serializes_camel_case_request_id(self):
        event = KnowledgeEvent(
            type="delta",
            content="检查冷却风机。",
            request_id="diagnosis-2",
        )

        payload = event.model_dump(by_alias=True, exclude_none=True)
        self.assertEqual(payload["requestId"], "diagnosis-2")

    def test_route_event_is_supported(self):
        event = KnowledgeEvent(type="route", content="direct")

        self.assertEqual(event.type, "route")


if __name__ == "__main__":
    unittest.main()
