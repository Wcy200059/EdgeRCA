import unittest

from edge_rca.note_audit import review_note


class NoteAuditTests(unittest.TestCase):
    def test_p95_and_total_count_confusion_is_flagged(self):
        facts = [{"current": {"count": 25366, "p95": 22, "max": 2036}}]
        warnings = review_note("出现25366次慢调用，P95为2036ms", facts)
        self.assertEqual(len(warnings), 2)
        self.assertEqual(review_note("共25366次调用，P95为22ms，最大2036ms", facts), [])

    def test_average_not_available_and_unknown_unit_conversion(self):
        self.assertTrue(review_note("平均延迟为0ms", [{"current": {"p50": 0}}]))
        self.assertTrue(review_note("可用内存2.83GB", [{"unit": "source_unit_unverified"}]))
        self.assertEqual(review_note("单位未知，不能换算GB", []), [])

    def test_gc_templates_are_not_occurrence_counts(self):
        facts = [{"kind": "log", "category": "gc", "status": "available", "count": 3},
                 {"kind": "log", "category": "gc", "status": "available", "count": 1}]
        self.assertTrue(review_note("期间发生2次GC", facts))
        self.assertEqual(review_note("期间发生4次GC，共2类", facts), [])

    def test_rounding_percentiles_and_different_scopes_are_accepted(self):
        facts = [{"current": {"p95": 98.46}, "first_trigger_values": {"p95_ms": 1087}}]
        self.assertEqual(review_note("P95为98.5ms，首次P95为1087ms", facts), [])

    def test_clean_numeric_checks_cannot_certify_causality(self):
        # Explicitly document the limit: arbitrary causal claims are not checked.
        self.assertEqual(review_note("已经确定网络是根因", []), [])


if __name__ == "__main__":
    unittest.main()
