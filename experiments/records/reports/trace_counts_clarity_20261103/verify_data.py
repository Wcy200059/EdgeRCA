"""Verify displayed provenance against frozen experiment artifacts; no model."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.trace_story import calls_at, visible_evidence
from tools.trace_pipeline import decode_reply, interpretation_origin, route_decision

OUT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'outputs/staged_full_20261102/manifest.json').read_text())
versions = {r['row']: r for r in manifest['sources']}
results = []
for file in sorted((ROOT / 'web/trace/data').glob('case_*.json')):
    data = json.loads(file.read_text()); case = Path(data['case']['path']); row = data['case']['row']
    source = versions[row]
    for relative, key in [('coordinator_1800/prediction.json','prediction_sha256'),('evaluation/result.json','evaluation_sha256')]:
        assert hashlib.sha256((case / relative).read_bytes()).hexdigest() == source[key]
    assert data['case']['source_sha256'] == source['inference_source_sha256']
    original_calls = {c['phase']: c for c in calls_at(case / 'coordinator_1800/calls.jsonl')}
    for round_ in data['story']['rounds']:
        saved = json.loads((case / f"coordinator_1800/input_{round_['index']}.json").read_text())
        snapshot = round_['snapshot']; original = original_calls[snapshot['phase']]
        assert visible_evidence(original, saved['binding']) == snapshot['evidence']
        assert {f['id'] for f in round_['archive']['facts'] if f['loaded']} == {e['id'] for e in snapshot['evidence']}
    facts = generated = defaults = 0
    for node in data['nodes']:
        original = json.loads((case / f"nodes_1800/device_{node['node']}/report.json").read_text())
        original_rows = {r['id']:r for r in original['template_rows']}
        for record in node['template_rows']:
            facts += 1
            assert record['fields'] == original_rows[record['id']]['fields']
            assert record['selection_source'] == interpretation_origin(original_rows[record['id']])
            generated += record['selection_source'] == 'program_generated'
            defaults += record.get('actual_decision',{}).get('source') == 'program_default'
            for a in record['appearances']:
                snapshot = next(r['snapshot'] for r in data['story']['rounds'] if r['index'] == a['round'])
                assert any(e['id']==a['id'] and e['source_id']==record['source_id'] for e in snapshot['evidence'])
        for batch in node['pipeline']['batches']:
            call = batch['attempts'][batch['accepted_attempt']]
            assert call['validation']['valid']
            for item in batch['items']:
                assert item['decision'] == route_decision(item['id'],decode_reply(call['raw_reply']),True)
    for node in data['nodes']:
        c = node['pipeline']['coverage']
        assert sum(c['raw_by_modality'].values()) == node['raw']
        assert sum(c['categories'].values()) == c['card_count'] == node['facts']
        assert sum(x['raw_samples'] for x in c['metric_names']) == c['raw_by_modality'].get('metric', 0)
        assert c['metric_name_count'] == c['represented_metric_names'] + c['unlisted_metric_names']
        assert c['omission_count_matches'], (row, node['node'])
        assert c['policy_verified_from_frozen_source'], (row, node['node'])
        for item in c['metric_names']:
            for material in item['materials']:
                assert material['catalogue_delivered'], (row, node['node'], material['id'])
    assert facts == data['flow']['facts']
    for source in data['pipeline']['metric_sources'].values():
        assert source['shown'] == len(source['samples'])
        assert source['omitted'] == source['total'] - source['shown']
        assert all(s['raw'].get('kpi_name') == source['metric'] for s in source['samples'])
    global_records = json.loads((case / 'frames/c1800/global_observations.json').read_text())
    by_id = {f['id']:f for f in global_records['facts']}
    initial = data['pipeline']['initial_archive']
    for item in data['pipeline']['global']['items']:
        assert item['fact'] == by_id[item['source_fact_id']]['tool_fact']
        assert item['kept'] == any(f['id']==item['id'] for f in initial['facts'])
    results.append({'row':row,'source_matches_frozen_experiment':True,'local_facts':facts,
                    'program_generated_interpretations':generated,'background_by_protocol_default':defaults,
                    'metric_names_in_local_records':sum(len(v) for v in data['pipeline']['metric_catalogues'].values()),
                    'metric_sample_sets':len(data['pipeline']['metric_sources']),
                    'linked_raw_records':len(data['pipeline']['source_records']),
                    'global_full_records':len(data['pipeline']['global']['items']),
                    'steps':len(data['pipeline']['steps']),'scores':data['pipeline']['scores']})
h=hashlib.sha256()
for p in sorted((ROOT/'edge_rca').glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
expected=json.loads((OUT/'before_manifest.json').read_text())['inference_source_sha256']
assert h.hexdigest()==expected
report={'cases':results,'inference_source_unchanged':True,'inference_source_sha256':h.hexdigest(),
        'original_predictions_and_evaluations_unchanged':True,'new_model_calls':0,'new_gpu_jobs':0}
(OUT/'data_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
