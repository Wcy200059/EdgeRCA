"""Protect the viewer's distinction between model choices and program artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tools.trace_pipeline import collect_record_ids, enrich_pipeline, fact_category, followup_calls, interpretation_origin, metric_inventory, metric_sources, node_coverage, route_decision, snapshot_archive


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


class PipelineTraceTests(unittest.TestCase):
    def test_kpi_series_and_mixed_materials_reconcile_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)
            facts = [
                {'id': 'f1', 'kind': 'alarm_history', 'signal': 'CPU'},
                {'id': 'f2', 'metric': 'Memory', 'selected_because': ['control']},
                {'id': 'f3', 'kind': 'alarm_history', 'signal': 'maximum_call_duration'},
                {'id': 'f4', 'kind': 'http_access', 'status': '200', 'count': 100},
                {'id': 'f5', 'kind': 'log', 'status': 'no_arrived_records_in_scope'},
                {'id': 'f6', 'kind': 'missing', 'modality': 'trace'},
                {'id': 'f7', 'kind': 'latency_over_analysis_window'},
                {'id': 'f8', 'kind': 'observed_connection', 'from': 'A', 'to': 'B'},
            ]
            packet = {'facts': facts, 'scope': {'raw_rows': {'metric': 7, 'trace': 100},
                                                'cpu_only_metric_count': 1},
                      'catalog': {'f1': {'metric': {'reference': {'count': 0}}},
                                  'f2': {'cpu_fact': {'reference': {'count': 0}}},
                                  'f3': {'alarm_view': {'modality': 'trace'}}}}
            save(case / 'frames/c1800/device_A.json', packet)
            node = {'node': 'A', 'template_rows': [
                {'id': 's0_f1', 'actual_decision': {'action': 'report'}},
                {'id': 's0_f2', 'actual_decision': {'action': 'background'}}]}
            counts = node_coverage(case, node, {'CPU': 3, 'Memory': 3, 'Disk': 1})
            self.assertEqual((counts['metric_name_count'], counts['represented_metric_names'],
                              counts['unlisted_metric_names'], counts['card_count']), (3, 2, 1, 8))
            self.assertEqual(counts['categories']['availability'], 2)
            self.assertEqual(counts['categories']['call_alarm'], 1)
            self.assertEqual(sum(counts['categories'].values()), 8)
            self.assertTrue(counts['omission_count_matches'])
            omitted = next(n for n in counts['metric_names'] if n['name'] == 'Disk')
            self.assertEqual(omitted['raw_samples'], 1)
            self.assertEqual(omitted['materials'], [])
            self.assertEqual(counts['metric_names'][1]['materials'][0]['reasons'], ['control'])
            counts = node_coverage(case, node, {'CPU': 3, 'Memory': 3})
            self.assertFalse(counts['omission_count_matches'])

    def test_metric_inventory_counts_samples_without_inventing_kpi_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'frames').mkdir()
            connection = sqlite3.connect(root / 'frames/inbox.sqlite')
            connection.execute('CREATE TABLE events (node_id TEXT, modality TEXT, payload TEXT)')
            for modality, raw in [('metric', {'kpi_name': 'CPU', 'value': 1}),
                                  ('metric', {'kpi_name': 'CPU', 'value': 2}),
                                  ('metric', {'kpi_name': 'Disk', 'value': 9}),
                                  ('trace', {'kpi_name': 'not-a-metric'})]:
                connection.execute('INSERT INTO events VALUES (?,?,?)',
                                   ('device:A', modality, json.dumps({'raw': raw})))
            connection.commit(); connection.close()
            self.assertEqual(metric_inventory(root), {'device:A': {'CPU': 2, 'Disk': 1}})
        self.assertEqual(fact_category({'kind': 'log', 'status': 'no_arrived_records_in_scope'}), 'availability')
        self.assertEqual(fact_category({'kind': 'http_access', 'count': 100}), 'log')

    def test_followup_retries_keep_the_saved_acceptance_status(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp); base = case / 'coordinator_1800'
            save(base / 'node_followup_0/manifest.json', {'files': ['step000/node_answer.json']})
            save(base / 'node_followup_0/step000/node_answer.json', {
                'answer': {'request_id': 'q1'}, 'pages': [{'page': 0, 'attempts': [
                    {'valid': False, 'error': 'unsupported reading'}, {'valid': True}]}]})
            rows = [{'phase': f'node_template/q1/0/{attempt}', 'messages': [], 'raw_reply': '{}'}
                    for attempt in range(2)]
            (base / 'node_calls_0.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            calls = followup_calls(case, 0)
            self.assertFalse(calls[0]['validation']['valid'])
            self.assertTrue(calls[1]['validation']['valid'])

    def test_catalogue_defer_does_not_become_a_model_written_interpretation(self):
        row = {'handling': 'catalogue_defer', 'model_selection': {'use': 'uncertain', 'reading': 'uncertain'}}
        self.assertEqual(interpretation_origin(row), 'program_generated')
        row['handling'] = 'model_checked'
        self.assertEqual(interpretation_origin(row), 'model')

    def test_sparse_page_default_is_not_an_explicit_model_background_choice(self):
        reply = {'report': ['f1'], 'defer': []}
        self.assertEqual(route_decision('f1', reply, True)['source'], 'model_explicit')
        self.assertEqual(route_decision('f2', reply, True), {'action': 'background', 'source': 'program_default'})
        self.assertEqual(route_decision('f2', reply, False)['action'], 'unprocessed')
        self.assertEqual(route_decision('f2', {'background': ['f2']}, True)['source'], 'model_explicit')

    def test_archive_and_index_do_not_make_a_fact_loaded(self):
        saved = {'view': {'evidence': [{'id': name, 'node_id': 'device:A', 'tool_fact': name}
                                      for name in ['e1', 'e2', 'e3']]},
                 'binding': {'e2': {'source_id': 'device:A@1/s0_f2'}}}
        snap = {'evidence': [{'id': 'e1'}], 'input': {'candidate_slate': [{'evidence_id': 'e2'}]}}
        result = snapshot_archive(saved, snap)
        self.assertTrue(result['facts'][0]['loaded'])
        self.assertFalse(result['facts'][1]['loaded'])
        self.assertTrue(result['facts'][1]['index_mentioned'])
        self.assertFalse(result['facts'][2]['index_mentioned'])
        self.assertEqual(len(saved['view']['evidence']), 3)

    def test_source_lists_and_metric_sample_limits_keep_truthful_counts(self):
        self.assertEqual(collect_record_ids([{'evidence_ids': ['a', 'b']}, {'event_id': 'a'}]), ['a', 'b'])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'frames').mkdir()
            connection = sqlite3.connect(root / 'frames/inbox.sqlite')
            connection.execute('CREATE TABLE events (node_id TEXT, modality TEXT, observed_at_ms INTEGER, payload TEXT)')
            for i, value in enumerate(['1', 'NaN', '3']):
                event = {'raw': {'kpi_name': 'CPU', 'value': value}, 'source_file': 'metrics.csv', 'source_row': i}
                connection.execute("INSERT INTO events VALUES ('device:A','metric',?,?)", (i, json.dumps(event)))
            connection.commit(); connection.close()
            result = metric_sources(root, {'device:A': {'CPU'}}, max_samples=2)['device:A|CPU']
            self.assertEqual((result['total'], result['shown'], result['omitted'], result['invalid_values']), (3, 2, 1, 1))
            self.assertIsNone(result['samples'][1]['value'])

    def test_real_catalogue_choice_and_program_generated_reading_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp); node = case / 'nodes_1800/device_A'
            original = {'id': 's0_f1', 'subject': 'CPU', 'fields': {}, 'template': 'metric_change',
                        'model_selection': {'reading': 'variable', 'use': 'report', 'detail_deferred': True},
                        'reading_options': {'variable': '波动'}, 'raw_fact': {'id': 's0_f1'}}
            save(node / 'report.json', {'node_id': 'device:A', 'cutoff_ms': 1, 'template_rows': [original], 'overview_text': '程序概览'})
            save(node / 'manifest.json', {'files': ['step000/node_answer.json']})
            save(node / 'step000/node_answer.json', {'answer': {'request_id': 'request-A',
                 'catalogue_audit': {'pages': [{'page': 0, 'attempts': [{'valid': True}]}]}}})
            # Lists are legitimate catalog entries for aggregated logs.
            save(node / 'step000/input.json', {'catalog': {'f1': [{'evidence_ids': ['record-1']}]}})
            call = {'phase': 'node_catalogue/request-A/0/0', 'call': 1,
                    'messages': [{'role': 'system', 'content': '选择ID'}, {'role': 'user', 'content': json.dumps({
                        'node': 'device:A', 'pages': 1, 'catalogue': [{'id': 'f1', 'subject': 'CPU'}]})}],
                    'raw_reply': '{"report":["f1"],"defer":[]}'}
            (case / 'nodes_1800/calls.jsonl').write_text(json.dumps(call) + '\n')
            central = case / 'coordinator_1800'; central.mkdir()
            binding = {'e1': {'source_id': 'device:A@1/s0_f1', 'node_id': 'device:A', 'origin': 'node_template'},
                       'e2': {'source_id': 'global_1/b1', 'node_id': 'cluster:observations', 'origin': 'followup'}}
            saved = {'binding': binding, 'view': {'evidence': [{'id': 'e1', 'node_id': 'device:A', 'tool_fact': 'CPU'}]}}
            save(central / 'input_0.json', saved); save(central / 'last_input.json', saved)
            save(case / 'frames/c1800/global_observations.json', {'request_id': 'global_1',
                 'facts': [{'id': 'b1', 'tool_fact': 'business:X；成功率0%', 'time_marks': []}]})
            save(central / 'global_filter.json', {'filter': {'ids': [], 'fallback': False, 'attempts': [{'valid': True}]}})
            global_call = {'phase': 'case/1800/global/global_filter/0', 'messages': [
                {'role': 'user', 'content': json.dumps({'evidence': [['e2', 'cluster:observations',
                '{"subject":"business:X","observations":{"text_in_detail_pages":true}}', []]]})}],
                'raw_reply': '{"ids":[],"notes":"不保留"}'}
            (central / 'calls.jsonl').write_text(json.dumps(global_call) + '\n')
            snapshot = {'key': 'r0', 'evidence': [{'id': 'e1', 'source_id': 'device:A@1/s0_f1'}], 'input': {}}
            data = {'nodes': [{'node': 'A', 'template_rows': [{'id': 's0_f1', 'subject': 'CPU', 'template': 'metric_change'}]}],
                    'story': {'rounds': [{'index': 0, 'snapshot': snapshot, 'snapshots': [], 'reply': {'hypotheses': []}}],
                              'final_snapshots': [], 'executed': []}, 'central': {'final': {'reply': None}}}
            result = enrich_pipeline(case, deepcopy(data))
            row = result['nodes'][0]['template_rows'][0]
            self.assertEqual(row['selection_source'], 'program_generated')
            self.assertEqual(row['actual_decision']['source'], 'model_explicit')
            self.assertEqual(row['central_id'], 'e1')
            self.assertEqual(row['sample_record_ids'], ['record-1'])
            global_item = result['pipeline']['global']['items'][0]
            self.assertFalse(global_item['kept'])
            self.assertIn('成功率0%', global_item['fact'])
            self.assertIn('text_in_detail_pages', global_item['filter_views'][0]['fact'])


if __name__ == '__main__':
    unittest.main()
