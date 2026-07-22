import unittest

from app.schemas.knowledge import KnowledgeEvent, KnowledgeRequest


class KnowledgeSchemaTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
