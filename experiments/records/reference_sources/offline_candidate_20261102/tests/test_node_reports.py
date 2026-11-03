import unittest
from copy import deepcopy

from edge_rca.node_reports import pages, validate, ReportInbox, build_snapshot, messages


class NodeReportTests(unittest.TestCase):
    def test_paging_never_drops_tail_or_duplicates_ids(self):
        facts = [{"id": f"f{i}", "text": "已保存的原始事实"} for i in range(35)]
        chunks = pages({"facts": facts}, max_facts=8, max_chars=1000)
        self.assertEqual([f for chunk in chunks for f in chunk], facts)
        self.assertEqual(len(chunks), 5)

    def test_oversized_single_fact_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "never truncate"):
            pages({"facts": [{"id": "f1", "text": "长" * 1000}]}, max_chars=100)

    def test_silent_omission_is_rejected(self):
        reply = {"groups": [], "retain_local": ["f1"], "defer": [], "questions": []}
        with self.assertRaisesRegex(ValueError, "no disposition"):
            validate(reply, [{"id": "f1"}, {"id": "f2"}])

    def test_retained_and_unprocessed_are_distinct(self):
        reply = {"groups": [], "retain_local": ["f1"], "defer": ["f2"], "questions": []}
        self.assertEqual(validate(reply, [{"id": "f1"}, {"id": "f2"}]), reply)

    def test_late_or_duplicate_report_does_not_replace_newer_report(self):
        inbox = ReportInbox()
        report = {"case_id": "c1", "node_id": "device:A", "cutoff_ms": 30, "groups": ["new"]}
        self.assertTrue(inbox.receive(report))
        report["groups"][0] = "mutated"
        self.assertFalse(inbox.receive({**report, "cutoff_ms": 10}))
        self.assertFalse(inbox.receive(report))
        self.assertEqual(inbox.latest[("c1", "device:A")]["groups"], ["new"])
        self.assertEqual(len(inbox.history), 3)

    def test_other_node_or_case_is_not_overwritten(self):
        inbox = ReportInbox()
        for case, node in (("c1", "A"), ("c2", "A"), ("c1", "B")):
            inbox.receive({"case_id": case, "node_id": node, "cutoff_ms": 10})
        self.assertEqual(len(inbox.latest), 3)

    def test_tools_with_different_cutoffs_are_rejected(self):
        memory = {"overview": {"node_id": "A", "cutoff_ms": 20}}
        with self.assertRaisesRegex(ValueError, "cutoff"):
            build_snapshot(memory, {"node_id": "A", "cutoff_ms": 30}, {"cutoff_ms": 20})

    def test_merge_prompt_exposes_only_current_layer_ids_without_losing_private_provenance(self):
        packet = {"node_id": "A", "scope": {}, "limitations": []}
        facts = [{"id": "g1", "source_ids": ["f17"],
                  "cpu_constraints": [{"source_id": "f17", "signal": "cpu", "state": "rules_cleared"}]}]
        original = deepcopy(facts)
        prompt = messages(packet, facts, merge=True)
        self.assertNotIn("f17", str(prompt))
        self.assertIn("g1", str(prompt))
        self.assertIn("rules_cleared", str(prompt))
        self.assertEqual(facts, original)


if __name__ == "__main__":
    unittest.main()
