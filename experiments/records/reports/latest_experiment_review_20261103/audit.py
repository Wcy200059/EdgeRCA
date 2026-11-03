"""CPU-only audit of frozen latest predictions and saved model interactions."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from edge_rca.scoring import COMPONENT, REASON, TIME, evaluate
from tools.bank_combine import inference_source_hash
from tools.trace_story import payload_from_call

OUT = Path(__file__).resolve().parent
LATEST = ROOT / 'outputs/staged_full_20261102'
manifest = json.loads((LATEST / 'manifest.json').read_text())
aggregate = {str(t): json.loads((LATEST / f'scores_{t}s/summary.json').read_text()) for t in (60, 300)}
rows = []
source_versions = Counter()
handling = Counter()
selection_readings = Counter()
total_catalogue = Counter()
filter_counts = Counter()
round_sizes = []
followup_stats = Counter()
missing_query_examples = []
invalid_details = []
node_call_phases = Counter()
field_totals = {str(t): Counter() for t in (60, 300)}
task_stats = {str(t): defaultdict(lambda: Counter()) for t in (60, 300)}
case_group_stats = {str(t): defaultdict(lambda: Counter()) for t in (60, 300)}


def calls(path):
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


for record in manifest['sources']:
    row = record['row']
    case = Path(record['source'])
    result_path = case / 'evaluation/result.json'
    raw_result = result_path.read_bytes()
    result = json.loads(raw_result)
    prediction_path = case / 'coordinator_1800/prediction.json'
    prediction_raw = prediction_path.read_bytes()
    assert hashlib.sha256(raw_result).hexdigest() == record['evaluation_sha256']
    assert hashlib.sha256(prediction_raw).hexdigest() == record['prediction_sha256'] == result['prediction_sha256']
    assert json.loads(prediction_raw) == result['prediction']
    version = inference_source_hash(case)
    assert version == record['inference_source_sha256']
    source_versions[version] += 1
    components, reasons, times = (pattern.findall(result['scoring_points']) for pattern in (COMPONENT, REASON, TIME))
    fields = {'component': components, 'reason': reasons, 'time': times}
    failures = max(map(len, fields.values()))
    group = '+'.join(key for key, items in fields.items() if items)
    scores = {}
    for tolerance in (60, 300):
        passed, failed, score = evaluate(prediction_raw.decode(), result['scoring_points'], time_tolerance_seconds=tolerance)
        saved = next(item for item in aggregate[str(tolerance)]['rows'] if item['row'] == row)
        assert score == saved['score']
        scores[str(tolerance)] = score
        for name, values in fields.items():
            accepted = Counter(passed)
            matched = sum(min(count, accepted[value]) for value, count in Counter(values).items())
            field_totals[str(tolerance)][name + '_scored'] += len(values)
            field_totals[str(tolerance)][name + '_correct'] += matched
        for label, bucket in ((result['task_index'], task_stats), (group, case_group_stats)):
            counter = bucket[str(tolerance)][label]
            counter.update(n=1, full=int(score == 1), any=int(score > 0), score_sum=score)
            counter['invalid'] += not result['final_reply_valid']
    central = case / 'coordinator_1800'
    final = json.loads((central / 'final_answer.json').read_text())
    if not result['final_reply_valid']:
        invalid_details.append({'row': row, 'expected_failures': failures,
                                'attempts': final['attempts'], 'instruction': result['instruction']})
    initial = json.loads((central / 'input_0.json').read_text())
    unfiltered_globals = {key for key, value in initial['binding'].items()
                          if value['node_id'] == 'cluster:observations'}
    triage = json.loads((central / 'global_filter.json').read_text())['filter']
    kept = set(triage['ids'])
    archived = initial['view']['evidence']
    globals_shown = [fact for fact in archived if fact['node_id'] == 'cluster:observations']
    business_kept = sum(str(f['tool_fact']).startswith('business:') for f in globals_shown)
    filter_counts.update(cases=1, fallback=int(triage.get('fallback', False)), empty=int(not kept),
                         available=len(unfiltered_globals), kept=len(kept), business_kept=business_kept,
                         cases_zero_business=int(business_kept == 0))
    stats = {'row': row, 'task': result['task_index'], 'scored_fields': group,
             'failures': failures, 'scores': scores, 'valid_final': result['final_reply_valid'],
             'model_calls': result['model_calls'], 'generation_seconds': result['model_generation_seconds'],
             'initial_archive': len(archived), 'local_initial_archive': len(archived) - len(globals_shown),
             'global_available': len(unfiltered_globals), 'global_kept': len(kept), 'business_kept': business_kept,
             'node_facts': 0, 'synthetic_readings': 0, 'catalogue_invalid_pages': 0,
             'context_rounds': [], 'followup_queries': []}
    literal_catalogue = defaultdict(set)
    for call in calls(case / 'nodes_1800/calls.jsonl'):
        node_call_phases[call['phase'].split('/', 1)[0]] += 1
        payload = payload_from_call(call)
        if 'catalogue' in payload:
            literal_catalogue[payload['node']].update(item['id'] for item in payload['catalogue'])
    stats['actual_catalogue_entries'] = sum(map(len, literal_catalogue.values()))
    for file in (case / 'nodes_1800').glob('device_*/step*/node_answer.json'):
        answer = json.loads(file.read_text())['answer']
        all_rows = answer.get('template_rows', [])
        stats['node_facts'] += len(all_rows)
        for item in all_rows:
            handling[item['handling']] += 1
            pick = item.get('model_selection') or {}
            if pick.get('detail_deferred'):
                stats['synthetic_readings'] += 1
                selection_readings[pick['reading']] += 1
        audit = answer.get('catalogue_audit') or {}
        for page in audit.get('pages', []):
            if not page.get('attempts') or not page['attempts'][-1].get('valid'):
                stats['catalogue_invalid_pages'] += 1
            total_catalogue['pages'] += 1
    for call in calls(central / 'calls.jsonl'):
        phase = call['phase']
        payload = payload_from_call(call)
        if 'global_filter/' in phase:
            literal = {f[0] for f in payload.get('evidence', [])}
            filter_counts['selected_but_not_in_literal_evidence'] += len(kept - literal)
            filter_counts['filter_calls'] += 1
            continue
        if '/round' not in phase:
            continue
        listed = payload.get('evidence', [])
        entry = {'phase': phase, 'tokens': call.get('input_tokens'), 'loaded': len(listed),
                 'device_nodes': len({f[1] for f in listed if f[1].startswith('device:')}),
                 'global_loaded': sum(f[1] == 'cluster:observations' for f in listed)}
        round_sizes.append(entry)
        stats['context_rounds'].append(entry)
    for path in (central / 'queries').glob('*/packet.json'):
        packet = json.loads(path.read_text())
        entry = {'node': packet['node_id'], 'tools': packet['tools_executed'], 'missing': packet.get('missing_metrics', []),
                 'focus': packet.get('request', {}).get('focus', '')}
        stats['followup_queries'].append(entry)
        followup_stats['total'] += 1
        if entry['missing']:
            followup_stats['with_missing_names'] += 1
            followup_stats['missing_names'] += len(entry['missing'])
            if len(missing_query_examples) < 15:
                missing_query_examples.append({'row': row, **entry})
        for tool in entry['tools']:
            followup_stats[tool] += 1
    complete = json.loads((central / 'complete.json').read_text())
    stats['executed_central_queries'] = complete['executed_central_queries']
    stats['global_filter_fallback'] = triage.get('fallback', False)
    rows.append(stats)


def distribution(values):
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values),
            'max': max(values), 'mean': statistics.mean(values)} if values else {}


comparisons = {}
for title, jobs in [('original_154057_154058', (154057, 154058)), ('previous_154464_154465', (154464, 154465))]:
    common, outcomes = [], {str(t): Counter() for t in (60, 300)}
    for item in rows:
        row = item['row']
        case = ROOT / f'outputs/bank_case_{jobs[0] if row < 46 else jobs[1]}_{row}'
        path = case / 'evaluation/result.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        raw = (case / 'coordinator_1800/prediction.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == result['prediction_sha256']
        sample = {'row': row, 'before': {}, 'after': item['scores']}
        for t in (60, 300):
            before = evaluate(raw.decode(), result['scoring_points'], time_tolerance_seconds=t)[2]
            after = item['scores'][str(t)]
            sample['before'][str(t)] = before
            count = outcomes[str(t)]
            count['n'] += 1
            count['before_score_sum'] += before
            count['after_score_sum'] += after
            count['before_full'] += before == 1
            count['after_full'] += after == 1
            count['improved'] += after > before
            count['regressed'] += after < before
            count['unchanged'] += after == before
            count['lost_full'] += before == 1 and after < 1
            count['new_full'] += before < 1 and after == 1
        common.append(sample)
    comparisons[title] = {'counts': outcomes, 'rows': common}

summary = {'checked_at': datetime.now(timezone.utc).isoformat(), 'arrays': [154750,154751],
           'source_versions': dict(source_versions), 'integrity_verified': len(rows),
           'score_summary': {t: {k:v for k,v in d.items() if k != 'rows'} for t,d in aggregate.items()},
           'field_totals': field_totals, 'tasks': task_stats, 'scored_groups': case_group_stats,
           'invalid_finals': invalid_details, 'initial_node_handling': dict(handling),
           'program_generated_reading_counts': dict(selection_readings), 'initial_node_calls': dict(node_call_phases),
           'catalogue': dict(total_catalogue), 'global_filter': dict(filter_counts),
           'followups': dict(followup_stats), 'missing_query_examples': missing_query_examples,
           'distributions': {key: distribution([r[key] for r in rows]) for key in
                             ['model_calls','generation_seconds','node_facts','actual_catalogue_entries','initial_archive',
                              'local_initial_archive','synthetic_readings','catalogue_invalid_pages']},
           'context_distributions': {key: distribution([r[key] for r in round_sizes]) for key in
                                    ['tokens','loaded','device_nodes','global_loaded']},
           'comparisons': comparisons, 'rows': rows,
           'scope': 'Saved end-to-end runs only; no new model inference or edits to original predictions.'}
(OUT / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:summary[k] for k in ['integrity_verified','source_versions','field_totals','tasks','scored_groups',
                                      'initial_node_handling','program_generated_reading_counts','global_filter','followups',
                                      'distributions','context_distributions']}, ensure_ascii=False, indent=2))
print(json.dumps({k:v['counts'] for k,v in comparisons.items()}, ensure_ascii=False, indent=2))
