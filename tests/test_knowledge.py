import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.knowledge import (
    KnowledgeService,
    is_contextual_follow_up,
    parse_markdown,
)


SAMPLE_KNOWLEDGE = """# 测试知识库

## 第三章 | 转辙机

### 3.2 故障模式

| 条目ID | 故障名称 | 故障现象 | 故障根因 |
|--------|---------|---------|---------|
| SW-021 | 锁闭装置磨损 | 锁闭不到位需重复动作 | 长期机械磨损 |
| SW-022 | 接点接触电阻增大 | 表示信号时有时无 | 接点氧化 |

## 第六章 | 牵引电机

| 条目ID | 故障名称 | 故障现象 |
|--------|---------|---------|
| MOT-001 | 定子绕组匝间短路 | 三相电流不平衡 |
| MOT-007 | 冷却风机故障 | 牵引电机温度异常升高 |
"""


class KnowledgeParserTest(unittest.TestCase):
    def test_parses_table_rows_with_source_metadata(self):
        chunks = parse_markdown(SAMPLE_KNOWLEDGE)

        sw_021 = next(chunk for chunk in chunks if chunk.entry_id == "SW-021")
        self.assertEqual(sw_021.chapter, "第三章 | 转辙机")
        self.assertEqual(sw_021.section, "3.2 故障模式")
        self.assertIn("锁闭不到位", sw_021.content)


class KnowledgeSearchTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name) / "knowledge.md"
        self.path.write_text(SAMPLE_KNOWLEDGE, encoding="utf-8")
        self.service = KnowledgeService(self.path, top_k=3)
        self.service.reload_if_changed(force=True)

    def tearDown(self):
        self.directory.cleanup()

    def test_exact_entry_id_is_ranked_first(self):
        results = self.service.search("请解释 SW-021")

        self.assertEqual(results[0].chunk.entry_id, "SW-021")

    def test_chinese_symptom_retrieves_matching_fault(self):
        results = self.service.search("转辙机锁闭不到位需要重复动作是什么原因")

        self.assertEqual(results[0].chunk.entry_id, "SW-021")

    def test_unrelated_question_returns_no_result(self):
        self.assertEqual(self.service.search("今天上海天气如何"), [])

    def test_info_reports_loaded_document(self):
        info = self.service.info()

        self.assertTrue(info["ready"])
        self.assertEqual(info["document"], "knowledge.md")
        self.assertEqual(info["chunkCount"], 4)

    def test_motor_temperature_query_retrieves_motor_fault(self):
        results = self.service.search("牵引电机温度异常如何排查")

        self.assertEqual(results[0].chunk.entry_id, "MOT-007")


class KnowledgeConversationTest(unittest.TestCase):
    def test_new_fault_topic_does_not_reuse_previous_context(self):
        self.assertFalse(is_contextual_follow_up("牵引电机温度异常如何排查"))

    def test_explicit_follow_up_reuses_previous_context(self):
        self.assertTrue(is_contextual_follow_up("那应该先检查哪里？"))
        self.assertTrue(is_contextual_follow_up("上述故障还需要测哪些参数？"))
