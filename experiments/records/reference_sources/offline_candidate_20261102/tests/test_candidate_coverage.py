"""CPU check for the neutral per-candidate modality coverage index."""

import unittest

from edge_rca.coordination import candidate_coverage


def row(subject, template="metric", use=None):
    return {"subject": subject, "template": template, "fields": {},
            "model_selection": {"use": use} if use else None}


class CandidateCoverageTests(unittest.TestCase):
    def test_each_public_candidate_has_a_row_and_network_is_counted(self):
        packet = {"system_context": {"root_cause_components": ["MG01", "Redis02"]},
                  "node_status": [{"node_id": "device:MG01", "status": "received"},
                                  {"node_id": "device:Redis02", "status": "received"}]}
        reports = {"device:MG01": {"template_rows": [
                       row("OSLinux-NETWORK_NET_RXDropped"), row("OSLinux-CPU_CPU_CPUCpuUtil")]},
                   "device:Redis02": {"template_rows": [row("OSLinux-MEMORY_MEMUsedMemPerc", use="report")]}}
        table = candidate_coverage(reports, packet)
        self.assertEqual([item["component"] for item in table], ["MG01", "Redis02"])
        self.assertEqual(table[0]["network"], 1)
        self.assertEqual(table[0]["resource"], 1)
        self.assertEqual(table[1]["reported"], 1)
        self.assertEqual(table[0]["report_status"], "received")


if __name__ == "__main__":
    unittest.main()
