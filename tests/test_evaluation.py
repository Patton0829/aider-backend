import json
import unittest
from pathlib import Path

from app.evaluation import percentile, score_turn, summarize
from app.services.knowledge import parse_markdown


class EvaluationTest(unittest.TestCase):
    def test_nearest_rank_percentiles_and_empty_samples(self):
        self.assertEqual(percentile([10, 20, 30, 40], .5), 20)
        self.assertEqual(percentile([10, 20, 30, 40], .95), 40)
        self.assertIsNone(percentile([], .5))

    def test_scores_recall_and_rejects_failed_identity_check(self):
        row = score_turn({
            "deltaCount": 3, "totalMs": 12, "route": "knowledge",
            "sources": [{"entryId": "SW-021"}],
            "assessment": {"answerable": True, "reason": "model_sufficient"},
            "validation": {"passed": False},
        }, {"route": "knowledge", "answerable": True, "entryIds": ["SW-021", "SW-022"]})
        self.assertEqual(row["recallAt5"], .5)
        self.assertFalse(row["passed"])

    def test_model_failure_does_not_count_as_correct_abstention(self):
        row = score_turn({"assessment": {"answerable": False, "reason": "check_unavailable"}}, {"answerable": False})
        self.assertFalse(row["checks"]["answerable"])

    def test_separates_followup_latency_and_counts_errors(self):
        report = summarize([
            {"route": "knowledge", "passed": True, "firstContentMs": 100, "totalMs": 300,
             "assessment": {"answerable": True}, "validation": {"passed": True},
             "metrics": {"followUp": False, "assessmentMs": 50}},
            {"route": "knowledge", "passed": True, "firstContentMs": 200, "totalMs": 400,
             "metrics": {"followUp": True}},
            {"error": "TimeoutError", "passed": False},
        ])
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["standaloneKnowledge"]["firstContentMs"]["p50"], 100)
        self.assertEqual(report["followupKnowledge"]["firstContentMs"]["p95"], 200)
        self.assertEqual(report["all"]["firstContentMs"]["mean"], 150)

    def test_failed_request_counts_as_zero_recall_and_incorrect_route(self):
        row = score_turn({}, {"route": "knowledge", "answerable": True, "entryIds": ["SW-021"]})
        row["error"] = "TimeoutError"
        report = summarize([row])
        self.assertEqual(report["meanRecallAt5"], 0)
        self.assertEqual(report["routeAccuracy"], 0)

    def test_all_dataset_entry_labels_exist_in_committed_knowledge(self):
        root = Path(__file__).resolve().parents[1]
        chunks = parse_markdown((root / "列车故障诊断经验知识库.md").read_text())
        entries = {c.entry_id for c in chunks}
        cases = json.loads((root / "evaluation/fault_diagnosis.json").read_text())["cases"]
        self.assertGreaterEqual(len(cases), 30)
        for case in cases:
            for turn in case["turns"]:
                self.assertTrue(set(turn.get("entryIds", [])).issubset(entries), case["id"])
