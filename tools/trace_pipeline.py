"""Read-only, loss-aware information handoffs for the offline trace viewer."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sqlite3

from .trace_story import calls_at, payload_from_call, read_json


def decode_reply(raw: str):
    text = re.sub(r'^```(?:json)?\s*', '', str(raw or '').strip(), flags=re.I)
    start = text.find('{')
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def route_decision(identifier: str, reply: dict | None, valid: bool) -> dict:
    """An omitted ID on an accepted sparse page is a program default."""
    if not valid or reply is None:
        return {'action': 'unprocessed', 'source': 'invalid_reply'}
    for key, action in [('report', 'report'), ('expand', 'report'), ('defer', 'defer'), ('background', 'background')]:
        if identifier in reply.get(key, []):
            return {'action': action, 'source': 'model_explicit', 'reply_key': key}
    return {'action': 'background', 'source': 'program_default'}


def interpretation_origin(row: dict) -> str:
    pick = row.get('model_selection') or {}
    if not pick:
        return 'none'
    if pick.get('detail_deferred') or row.get('handling') in ('catalogue_report', 'catalogue_defer'):
        return 'program_generated'
    return 'model'


def call_record(call: dict, validation=None) -> dict:
    return {'phase': call.get('phase'), 'number': call.get('call'),
            'input_tokens': call.get('input_tokens'), 'output_tokens': call.get('output_tokens'),
            'seconds': call.get('seconds'), 'hit_output_limit': call.get('hit_output_limit'),
            'messages': call.get('messages', []), 'input': payload_from_call(call),
            'raw_reply': call.get('raw_reply', ''), 'parsed': decode_reply(call.get('raw_reply', '')),
            'validation': validation}


def collect_record_ids(value) -> list[str]:
    ids = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('evidence_ids', 'record_ids') and isinstance(item, list):
                ids.extend(i for i in item if isinstance(i, str))
            elif key == 'event_id' and isinstance(item, str):
                ids.append(item)
            elif isinstance(item, (dict, list)):
                ids.extend(collect_record_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.extend(collect_record_ids(item))
    return list(dict.fromkeys(ids))


def metric_sources(case: Path, wanted: dict[str, set[str]], max_samples=60) -> dict:
    """All represented metric names; bounded samples are explicitly counted."""
    result = {}
    path = case / 'frames/inbox.sqlite'
    if not path.exists() or not wanted:
        return result
    connection = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    try:
        for node, stamp, payload in connection.execute(
                "SELECT node_id, observed_at_ms, payload FROM events WHERE modality='metric' ORDER BY observed_at_ms"):
            if node not in wanted:
                continue
            event = json.loads(payload)
            raw = event.get('raw', {})
            metric = raw.get('kpi_name')
            if metric not in wanted[node]:
                continue
            key = node + '|' + metric
            item = result.setdefault(key, {'node_id': node, 'metric': metric, 'total': 0,
                                           'invalid_values': 0, 'samples': []})
            item['total'] += 1
            try:
                number = float(raw.get('value'))
                finite = math.isfinite(number)
            except (ValueError, TypeError):
                number, finite = None, False
            if not finite:
                item['invalid_values'] += 1
                number = None
            if len(item['samples']) < max_samples:
                item['samples'].append({'observed_ms': stamp, 'value': number, 'raw': raw,
                                        'event_id': event.get('event_id'), 'source_file': event.get('source_file'),
                                        'source_row': event.get('source_row')})
    finally:
        connection.close()
    for item in result.values():
        item['shown'] = len(item['samples'])
        item['omitted'] = item['total'] - item['shown']
    return result


def record_sources(case: Path, identifiers: set[str]) -> dict:
    """Resolve a few actual source IDs per observation without guessing lineage."""
    path = case / 'frames/inbox.sqlite'
    if not path.exists() or not identifiers:
        return {}
    result = {}
    connection = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    try:
        if 'event_id' not in {row[1] for row in connection.execute('PRAGMA table_info(events)')}:
            return {}
        ids = sorted(identifiers)
        for offset in range(0, len(ids), 400):
            batch = ids[offset:offset + 400]
            placeholders = ','.join('?' for _ in batch)
            for identifier, node, modality, observed, payload in connection.execute(
                    f'SELECT event_id,node_id,modality,observed_at_ms,payload FROM events WHERE event_id IN ({placeholders})', batch):
                event = json.loads(payload)
                result[identifier] = {'event_id': identifier, 'node_id': node, 'modality': modality,
                                      'observed_ms': observed, 'raw': event.get('raw', event),
                                      'source_file': event.get('source_file'), 'source_row': event.get('source_row')}
    finally:
        connection.close()
    return result


def metric_inventory(case: Path) -> dict[str, dict[str, int]]:
    """Count original samples by exact KPI name, before any model selection."""
    path = case / 'frames/inbox.sqlite'
    if not path.exists():
        return {}
    names = defaultdict(Counter)
    connection = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    try:
        for node, payload in connection.execute("SELECT node_id,payload FROM events WHERE modality='metric'"):
            metric = json.loads(payload).get('raw', {}).get('kpi_name')
            if isinstance(metric, str):
                names[node][metric] += 1
    finally:
        connection.close()
    return {node: dict(sorted(items.items())) for node, items in names.items()}


def metric_catalogues(case: Path) -> dict[str, list[str]]:
    return {node: list(items) for node, items in metric_inventory(case).items()}


def fact_category(fact: dict, source=None) -> str:
    """Classify saved materials without presenting logs or missing data as KPIs."""
    source = source if isinstance(source, dict) else {}
    kind = fact.get('kind')
    if kind == 'missing' or fact.get('status') == 'no_arrived_records_in_scope':
        return 'availability'
    if kind == 'alarm_history':
        modality = source.get('alarm_view', {}).get('modality')
        if source.get('metric') or fact.get('whole_prefix') or modality == 'metric':
            return 'kpi_alarm'
        return 'call_alarm' if modality == 'trace' else 'other_alarm'
    if fact.get('metric'):
        return 'kpi_summary'
    return {'log': 'log', 'http_access': 'log', 'latency_over_analysis_window': 'call_summary',
            'observed_connection': 'connection'}.get(kind, 'other')


def node_coverage(case: Path, node: dict, inventory: dict[str, int]) -> dict:
    """Reconcile source records, named time series, and the first 4B packet.

    Admission reasons are copied from saved packets. Names not represented are
    obtained by exact set difference, checked against the saved omission count.
    This does not rerun inference or invent per-metric rejection decisions.
    """
    packet = read_json(case / 'frames/c1800' / ('device_' + node['node'] + '.json'), {})
    if not packet:
        return {}
    categories = Counter()
    represented = defaultdict(list)
    details = []
    rows = {r['id']: r for r in node.get('template_rows', [])}
    for fact in packet.get('facts', []):
        identifier = fact['id']
        source = packet.get('catalog', {}).get(identifier, {})
        category = fact_category(fact, source)
        categories[category] += 1
        local_id = 's0_' + identifier
        details.append({'id': local_id, 'category': category})
        if category not in ('kpi_alarm', 'kpi_summary'):
            continue
        name = fact.get('signal') or fact.get('metric')
        metric = source.get('metric') if category == 'kpi_alarm' else (source or {}).get('cpu_fact')
        metric = metric or {}
        represented[name].append({
            'id': local_id, 'reasons': ['alarm_history'] if category == 'kpi_alarm' else fact.get('selected_because', []),
            'reference_count': metric.get('reference', {}).get('count'),
            'current_count': metric.get('current', {}).get('count'),
            'catalogue_delivered': rows.get(local_id, {}).get('actual_decision') is not None})
    names = [{'name': name, 'raw_samples': count, 'included': name in represented,
              'materials': represented.get(name, [])} for name, count in inventory.items()]
    not_represented = sum(not item['included'] for item in names)
    saved_count = packet.get('scope', {}).get('cpu_only_metric_count')
    frozen = case / 'source/edge_rca/compact_tools.py'
    code = frozen.read_text() if frozen.exists() else ''
    policy_verified = all(fragment in code for fragment in (
        'ratio >= 1.5 or 0 <= ratio <= .5', 'enough_samples_for_description',
        'if not (f["name"] in alerted_names or changed or quality or control)'))
    return {'raw_by_modality': packet.get('scope', {}).get('raw_rows', {}),
            'card_count': len(packet.get('facts', [])), 'categories': dict(categories), 'cards': details,
            'metric_names': names, 'metric_name_count': len(names),
            'represented_metric_names': sum(item['included'] for item in names),
            'unlisted_metric_names': not_represented, 'saved_unlisted_metric_observations': saved_count,
            'omission_count_matches': saved_count == not_represented,
            'policy_verified_from_frozen_source': policy_verified,
            'source_files': ['frames/c1800/device_' + node['node'] + '.json',
                             'source/edge_rca/node_reports.py', 'source/edge_rca/compact_tools.py']}


def node_materials(case: Path, node: dict) -> dict:
    directory = case / 'nodes_1800' / ('device_' + node['node'])
    report = read_json(directory / 'report.json', {})
    manifest = read_json(directory / 'manifest.json', {})
    limitations = []
    for filename in manifest.get('files', []):
        packet = read_json(directory / Path(filename).parent / 'input.json', {})
        limitations.extend(packet.get('limitations', []))
    return {'cpu_only_metric_ids': report.get('cpu_only_metric_ids', []),
            'limitations': list(dict.fromkeys(limitations))}


def followup_calls(case: Path, round_index: int) -> list[dict]:
    base = case / 'coordinator_1800'
    directory = base / f'node_followup_{round_index}'
    manifest = read_json(directory / 'manifest.json', {})
    validations = {}
    for filename in manifest.get('files', []):
        saved = read_json(directory / filename, {})
        answer = saved.get('answer', {})
        request_id = answer.get('request_id')
        for prefix, pages in [('node_template', saved.get('pages', [])),
                              ('node_catalogue', (answer.get('catalogue_audit') or {}).get('pages', []))]:
            for page in pages:
                for attempt, validation in enumerate(page.get('attempts', [])):
                    validations[f"{prefix}/{request_id}/{page['page']}/{attempt}"] = validation
    return [call_record(call, validations.get(call.get('phase')))
            for call in calls_at(base / f'node_calls_{round_index}.jsonl')]


def packing_policy(case: Path) -> dict:
    memory_file = case / 'source/edge_rca/evidence_memory.py'
    runtime_file = case / 'source/edge_rca/case_runtime.py'
    memory = memory_file.read_text() if memory_file.exists() else ''
    runtime = runtime_file.read_text() if runtime_file.exists() else ''
    limit = re.search(r'LocalModel\(coordinator_model,[^\n]*max_context=(\d+)', runtime)
    reserve = re.search(r'tokens\s*\+\s*output_tokens\s*\+\s*(\d+)\s*<=', memory)
    return {'context_limit': int(limit.group(1)) if limit else None,
            'repair_reserve': int(reserve.group(1)) if reserve else None,
            'cross_node_priority': 'per_node[f["node_id"]] == 1' in memory,
            'explicit_pages': 'f["id"] in pages' in memory,
            'deduplicates': 'duplicate_of' in memory,
            'source_file': 'source/edge_rca/evidence_memory.py'}


def snapshot_archive(saved: dict, snapshot: dict | None) -> dict:
    view = saved.get('view', {})
    shown = {e['id'] for e in (snapshot or {}).get('evidence', [])}
    index_ids = set()
    payload = (snapshot or {}).get('input', {})
    for item in payload.get('candidate_slate', []):
        if item.get('evidence_id'):
            index_ids.add(item['evidence_id'])
    for item in payload.get('evidence_directory', []):
        if isinstance(item, list) and len(item) > 3 and isinstance(item[3], list):
            index_ids.update(item[3])
    binding = saved.get('binding', {})
    facts = []
    for fact in view.get('evidence', []):
        identifier = fact['id']
        origin = binding.get(identifier, {})
        facts.append({'id': identifier, 'node_id': fact.get('node_id'),
                      'source_id': origin.get('source_id'), 'origin': origin.get('origin'),
                      'fact': fact.get('tool_fact'), 'time_marks': fact.get('time_marks', []),
                      'fields': fact.get('detail_fields', fact.get('template_payload', {}).get('fields')),
                      'loaded': identifier in shown, 'index_mentioned': identifier in index_ids})
    return {'facts': facts, 'count': len(facts), 'loaded_count': len(shown),
            'node_summaries': view.get('node_summaries', []),
            'node_status': view.get('node_status', []), 'candidate_coverage': view.get('candidate_coverage', []),
            'candidate_slate': view.get('candidate_slate', []), 'binding': binding}


def enrich_pipeline(case: Path, data: dict) -> dict:
    central = case / 'coordinator_1800'
    inventory = metric_inventory(case)
    central_calls = calls_at(central / 'calls.jsonl')
    node_calls = calls_at(case / 'nodes_1800/calls.jsonl')
    initial = read_json(central / 'input_0.json', {})
    initial_binding = initial.get('binding', {})
    initial_reverse = {v.get('source_id'): key for key, v in initial_binding.items()}
    rounds = data.get('story', {}).get('rounds', [])
    final_snapshots = data.get('story', {}).get('final_snapshots', [])
    citations = defaultdict(list)
    appearances = defaultdict(list)
    for item in rounds:
        snapshot = item.get('snapshot') or {}
        saved = read_json(central / f"input_{item['index']}.json", {})
        item['archive'] = snapshot_archive(saved, snapshot)
        for attempt, call in zip(item.get('snapshots', []),
                                 [c for c in central_calls if re.match(rf"case/\d+/round{item['index']}/\d+$", c.get('phase', ''))]):
            attempt['messages'] = call.get('messages', [])
        for evidence in snapshot.get('evidence', []):
            if evidence.get('source_id'):
                appearances[evidence['source_id']].append({'round': item['index'], 'id': evidence['id'], 'snapshot': snapshot['key']})
        for hypothesis in (item.get('reply') or {}).get('hypotheses', []):
            for field, role in [('support_ids', 'support'), ('counter_ids', 'counter')]:
                for identifier in hypothesis.get(field, []):
                    source = saved.get('binding', {}).get(identifier, {}).get('source_id')
                    if source:
                        citations[source].append({'round': item['index'], 'id': identifier, 'role': role,
                                                   'component': hypothesis.get('component'), 'reason': hypothesis.get('reason'),
                                                   'snapshot': snapshot.get('key')})
    last_saved = read_json(central / 'last_input.json', {})
    for snap, call in zip(final_snapshots, [c for c in central_calls if c.get('phase', '').startswith('final_commit/')]):
        snap['messages'] = call.get('messages', [])
    final_reply = data.get('central', {}).get('final', {}).get('reply') or {}
    for fault in final_reply.get('faults', []):
        for identifier in fault.get('evidence_ids', []):
            source = last_saved.get('binding', {}).get(identifier, {}).get('source_id')
            if source:
                citations[source].append({'round': 'final', 'id': identifier, 'role': 'final',
                                           'component': fault.get('component'), 'reason': fault.get('reason'),
                                           'snapshot': final_snapshots[-1]['key'] if final_snapshots else None})
    batches_by_node = defaultdict(list)
    call_lookup = defaultdict(list)
    for call in node_calls:
        payload = payload_from_call(call)
        node = str(payload.get('node', '')).split(':')[-1]
        call_lookup[node].append(call)
    wanted = defaultdict(set)
    for node in data.get('nodes', []):
        node_name = node['node']
        directory = case / 'nodes_1800' / ('device_' + node_name)
        report = read_json(directory / 'report.json', {})
        originals = {r['id']: r for r in report.get('template_rows', [])}
        step_info = {}
        manifest = read_json(directory / 'manifest.json', {})
        for file in manifest.get('files', []):
            step = re.match(r'step(\d+)/', file)
            if not step:
                continue
            answer = read_json(directory / file, {}).get('answer', {})
            packet = read_json(directory / Path(file).parent / 'input.json', {})
            step_info[int(step.group(1))] = {'answer': answer, 'packet': packet, 'file': file}
        batches = []
        grouped = defaultdict(list)
        for call in call_lookup[node_name]:
            parts = str(call.get('phase', '')).split('/')
            if len(parts) >= 4 and parts[0] == 'node_catalogue':
                grouped[(parts[1], int(parts[2]))].append(call)
        decisions_for_row = {}
        for (request_id, page_number), calls in grouped.items():
            info = next((v for v in step_info.values() if v['answer'].get('request_id') == request_id), {})
            page = next((p for p in (info.get('answer', {}).get('catalogue_audit') or {}).get('pages', [])
                         if p.get('page') == page_number), {})
            validations = page.get('attempts', [])
            attempts = [call_record(call, validations[i] if i < len(validations) else None)
                        for i, call in enumerate(calls)]
            accepted = next((i for i in reversed(range(len(attempts)))
                             if (attempts[i].get('validation') or {}).get('valid')), None)
            effective = attempts[accepted] if accepted is not None else attempts[-1]
            valid = accepted is not None
            step_number = next((number for number, v in step_info.items()
                                if v['answer'].get('request_id') == request_id), 0)
            inputs = effective['input'].get('catalogue', [])
            batch = {'key': f'{node_name}/{request_id}/{page_number}', 'page': page_number + 1,
                     'pages': effective['input'].get('pages'), 'attempts': attempts,
                     'accepted_attempt': accepted, 'valid': valid, 'items': []}
            for item in inputs:
                local = f"s{step_number}_{item['id']}"
                choice = route_decision(item['id'], effective['parsed'], valid)
                batch['items'].append({**deepcopy(item), 'local_id': local, 'decision': choice})
                decisions_for_row[local] = {**choice, 'batch_key': batch['key'], 'page': batch['page'],
                                            'input_id': item['id'], 'call': effective['number']}
            batches.append(batch)
        for row in node.get('template_rows', []):
            original = originals.get(row['id'], {})
            match = re.match(r's(\d+)_(.*)', row['id'])
            step_number, fact_id = (int(match.group(1)), match.group(2)) if match else (0, row['id'])
            info = step_info.get(step_number, {})
            source = info.get('packet', {}).get('catalog', {}).get(fact_id, {})
            alarm_view = (source.get('alarm_view') or {}) if isinstance(source, dict) else {}
            pick = original.get('model_selection') or {}
            row.update(raw_fact=original.get('raw_fact', {}),
                       material_category=fact_category(original.get('raw_fact', {}), source),
                       reading_options=original.get('reading_options', {}),
                       selection_source=interpretation_origin(original),
                       actual_decision=decisions_for_row.get(row['id']),
                       source_id=f"device:{node_name}@{report.get('cutoff_ms')}/{row['id']}",
                       source_record_ids=collect_record_ids(source),
                       rule_view={k: alarm_view.get(k) for k in
                                  ('active_rules', 'status', 'first_seen', 'last_trigger', 'last_change', 'state_changes')},
                       source_files={'packet': f"nodes_1800/device_{node_name}/step{step_number:03d}/input.json",
                                     'report': f"nodes_1800/device_{node_name}/report.json"})
            row['central_id'] = initial_reverse.get(row['source_id'])
            row['sample_record_ids'] = row['source_record_ids'][:3]
            row['appearances'] = appearances.get(row['source_id'], [])
            row['citations'] = citations.get(row['source_id'], [])
            if row.get('template') == 'metric_change':
                wanted['device:' + node_name].add(row['subject'])
                row['metric_source_key'] = 'device:' + node_name + '|' + row['subject']
        node['pipeline'] = {'batches': batches, 'summary': report.get('overview_text'),
                            'scope': report.get('scope', {}), 'questions': report.get('questions', []),
                            'pending_queries': report.get('pending_self_queries', []),
                            'counts': dict(Counter((r.get('actual_decision') or {}).get('action', 'unknown')
                                                   for r in node.get('template_rows', [])))}
        node['pipeline'].update(node_materials(case, node))
        node['pipeline']['coverage'] = node_coverage(case, node, inventory.get('device:' + node_name, {}))
    gf_saved = read_json(central / 'global_filter.json', {})
    gf = gf_saved.get('filter') or {}
    gf_calls = [c for c in central_calls if '/global_filter/' in c.get('phase', '')]
    gf_attempts = [call_record(c, gf.get('attempts', [])[i] if i < len(gf.get('attempts', [])) else None)
                   for i, c in enumerate(gf_calls)]
    full_globals = read_json(case / 'frames/c1800/global_observations.json', {})
    initial_ids = {f['id'] for f in initial.get('view', {}).get('evidence', [])}
    global_items = []
    for fact in full_globals.get('facts', []):
        source_id = f"{full_globals.get('request_id')}/{fact['id']}"
        identifier = initial_reverse.get(source_id)
        filter_views = []
        for attempt in gf_attempts:
            listed = next((r for r in attempt['input'].get('evidence', []) if r[0] == identifier), None)
            if listed:
                filter_views.append({'phase': attempt['phase'], 'fact': listed[2], 'time_marks': listed[3]})
        global_items.append({'id': identifier, 'source_id': source_id, 'source_fact_id': fact['id'],
                             'kind': 'business' if str(fact.get('tool_fact', '')).startswith('business:') else 'call',
                             'fact': fact.get('tool_fact'), 'time_marks': fact.get('time_marks', []),
                             'filter_views': filter_views, 'kept': identifier in initial_ids,
                             'appearances': appearances.get(source_id, []), 'citations': citations.get(source_id, [])})
    allowed_globals = {item['id'] for item in global_items if item['id']}
    for attempt in gf_attempts:
        picked = (attempt.get('parsed') or {}).get('ids', [])
        attempt['out_of_scope_ids'] = [i for i in picked if isinstance(i, str) and i not in allowed_globals]
    for query in data.get('story', {}).get('executed', []):
        packet = read_json(central / 'queries' / str(query['request_id']) / 'packet.json', {})
        query['tool_packet'] = {k: packet.get(k) for k in
                                ('scope', 'requested_scope', 'tools_executed', 'facts', 'missing_metrics',
                                 'available_metric_names', 'limitations')}
        query['packet_file'] = f"coordinator_1800/queries/{query['request_id']}/packet.json"
        query['actual_calls'] = followup_calls(case, query['round'])
        next_round = next((r for r in rounds if r['index'] == query['round'] + 1), None)
        next_saved = read_json(central / f"input_{query['round'] + 1}.json", {})
        query['returned_ids'] = [key for key, value in next_saved.get('binding', {}).items()
                                  if str(value.get('source_id', '')).startswith(str(query['request_id']) + '/')]
        citations_next = {i for h in ((next_round or {}).get('reply') or {}).get('hypotheses', [])
                          for i in h.get('support_ids', []) + h.get('counter_ids', [])}
        query['cited_in_next_round'] = [i for i in query['returned_ids'] if i in citations_next]
    steps = [
        {'kind': 'raw', 'stage': 0, 'actor': 'program', 'title': '保存原始记录，按节点归集'},
        {'kind': 'facts', 'stage': 1, 'actor': 'program', 'title': '计算节点观测与规则状态'},
        {'kind': 'batch', 'stage': 2, 'actor': '4b', 'title': '4B逐页决定上报、待查或背景'},
        {'kind': 'handoff', 'stage': 3, 'actor': 'program', 'title': '把节点报告交给中央'},
        {'kind': 'globals', 'stage': 3, 'actor': 'program', 'title': '接入独立的业务和调用统计'},
    ]
    if gf_attempts:
        steps.append({'kind': 'filter', 'stage': 4, 'actor': '9b', 'title': '9B筛选全局信息'})
    for item in rounds:
        index = item['index']
        steps.extend([
            {'kind': 'context', 'stage': 4, 'actor': 'program', 'round': index, 'title': f'第{index + 1}轮：装配9B实际输入'},
            {'kind': 'decision', 'stage': 4, 'actor': '9b', 'round': index, 'title': f'第{index + 1}轮：9B比较候选与证据'},
        ])
        query = next((q for q in data.get('story', {}).get('executed', []) if q['round'] == index), None)
        if query:
            steps.extend([
                {'kind': 'query', 'stage': 5, 'actor': 'program', 'round': index, 'title': f"执行补查：{query['node']}"},
                {'kind': 'return', 'stage': 5, 'actor': '4b', 'round': index, 'title': f"4B答复并追加材料：{query['node']}"},
            ])
        page = read_json(central / f'page_request_{index}.json', None)
        if page:
            steps.append({'kind': 'pages', 'stage': 5, 'actor': 'program', 'round': index,
                          'title': '加载已有证据详情页', 'request': page})
    steps.extend([{'kind': 'final', 'stage': 6, 'actor': '9b', 'title': '9B给出并冻结最终答案'},
                  {'kind': 'score', 'stage': 6, 'actor': 'program', 'title': '独立评分：核对冻结后的答案'}])
    sample_ids = {identifier for node in data.get('nodes', [])
                  for row in node.get('template_rows', []) for identifier in row.get('sample_record_ids', [])}
    scores = []
    prediction_file = central / 'prediction.json'
    evaluation = read_json(case / 'evaluation/result.json', {})
    if prediction_file.exists() and evaluation.get('scoring_points'):
        from edge_rca.scoring import evaluate
        for tolerance in (60, 300):
            passed, failed, score = evaluate(prediction_file.read_text(), evaluation['scoring_points'],
                                             time_tolerance_seconds=tolerance)
            scores.append({'tolerance': tolerance, 'score': score, 'passed': passed, 'failed': failed})
    data['pipeline'] = {'version': 1, 'steps': steps,
                        'metric_sources': metric_sources(case, wanted),
                        'source_records': record_sources(case, sample_ids), 'scores': scores,
                        'metric_catalogues': {node: list(names) for node, names in inventory.items()},
                        'packing_policy': packing_policy(case),
                        'global': {'items': global_items, 'attempts': gf_attempts, 'result': gf,
                                   'source_file': 'frames/c1800/global_observations.json'},
                        'initial_archive': snapshot_archive(initial, rounds[0].get('snapshot') if rounds else None),
                        'final_archive': snapshot_archive(last_saved, final_snapshots[-1] if final_snapshots else None),
                        'initial_counts': dict(Counter(r.get('selection_source') for n in data.get('nodes', [])
                                                       for r in n.get('template_rows', []))),
                        'evidence_contract': 'Only literal evidence messages and system excerpts count as loaded; indexes are separate.'}
    return data
