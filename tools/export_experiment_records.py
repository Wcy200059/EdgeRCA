"""Archive completed experiment evidence; never run models or copy raw databases."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

from edge_rca.scoring import evaluate
from .experiment_records import safe_path


TEXT_SUFFIXES = {'.json', '.jsonl', '.md', '.txt', '.log', '.err', '.py', '.sh', '.sbatch', '.patch', '.csv'}
SECRET = re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----')


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def source_fingerprint(directory: Path) -> str | None:
    files = sorted((directory / 'edge_rca').glob('*.py'))
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class DigestWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, chunk):
        self.digest.update(chunk)
        self.size += len(chunk)
        return self.stream.write(chunk)

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.size


class ArchiveWriter:
    def __init__(self, project: Path, output: Path, threshold=65536, level=6):
        self.project, self.output = project.resolve(), output.resolve()
        self.threshold, self.level = threshold, level
        self.counts = Counter()

    def store(self, source: Path, name: str, *, original_path: str | None = None) -> dict:
        if source.is_symlink() or not source.resolve().is_relative_to(self.project):
            raise ValueError(f'source must be a regular project file: {source}')
        before = source.stat()
        compressed = source.suffix == '.json' and before.st_size >= self.threshold
        stored_name = name + '.gz' if compressed else name
        destination = safe_path(self.output, stored_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest, tail, total = hashlib.sha256(), b'', 0
        with source.open('rb') as original, destination.open('xb') as target:
            sink = DigestWriter(target)
            packed = gzip.GzipFile(filename='', fileobj=sink, mode='wb',
                                   compresslevel=self.level, mtime=0) if compressed else sink
            try:
                while chunk := original.read(128 * 1024):
                    if source.suffix in TEXT_SUFFIXES and SECRET.search(tail + chunk):
                        raise ValueError(f'possible credential in {source}; review before publishing')
                    tail = chunk[-512:]
                    digest.update(chunk)
                    total += len(chunk)
                    packed.write(chunk)
            finally:
                if compressed:
                    packed.close()
            stored_digest, stored_size = sink.digest.hexdigest(), sink.size
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'completed record changed during export: {source}')
        if stored_size >= 50_000_000:
            raise ValueError(f'archive member unexpectedly large: {stored_name}')
        self.counts.update(files=1, raw_bytes=total, stored_bytes=stored_size,
                           gzip_files=int(compressed))
        return {'path': original_path or name, 'stored_path': stored_name,
                'source': source.resolve().relative_to(self.project).as_posix(),
                'compression': 'gzip' if compressed else 'none', 'raw_bytes': total,
                'stored_bytes': stored_size, 'sha256': digest.hexdigest(), 'stored_sha256': stored_digest}


def copy_tree(writer: ArchiveWriter, source: Path, destination: str) -> list[dict]:
    files = []
    for path in sorted(source.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.suffix in TEXT_SUFFIXES:
            if '__pycache__' in path.parts:
                continue
            relative = path.relative_to(source).as_posix()
            files.append(writer.store(path, destination + '/' + relative, original_path=relative))
    return files


def add_inference_source(writer: ArchiveWriter, directory: Path, sources: dict) -> str | None:
    fingerprint = source_fingerprint(directory)
    if fingerprint and fingerprint not in sources:
        files = []
        for path in sorted((directory / 'edge_rca').glob('*.py')):
            relative = 'edge_rca/' + path.name
            files.append(writer.store(path, f'sources/{fingerprint}/{relative}', original_path=relative))
        sources[fingerprint] = {'fingerprint': fingerprint, 'files': files,
                                'scope': 'frozen edge_rca inference modules'}
    return fingerprint


def case_files(case: Path, full: bool) -> list[Path]:
    if full:
        paths = []
        for name in ('nodes_1800', 'coordinator_1800', 'evaluation', 'frames/c1800'):
            paths.extend(path for path in (case / name).rglob('*')
                         if path.is_file() and not path.is_symlink() and path.suffix in TEXT_SUFFIXES)
        paths.extend(case / 'frames' / name for name in ('complete.json', 'public_question.json'))
    else:
        paths = [case / name for name in (
            'evaluation/result.json', 'evaluation/结果.md', 'coordinator_1800/prediction.json',
            'coordinator_1800/final_answer.json', 'frames/complete.json', 'frames/public_question.json')]
    return sorted(set(path for path in paths if path.is_file() and not path.is_symlink()))


def archive_run(writer: ArchiveWriter, manifest_path: Path, label: str, sources: dict, *, full: bool) -> dict:
    original = load(manifest_path)
    records = original['sources']
    expected = original['expected_rows']
    if len(records) != expected or {row['row'] for row in records} != set(range(expected)):
        raise ValueError(f'incomplete run manifest: {manifest_path}')
    cases, scores = [], []
    for number, record in enumerate(sorted(records, key=lambda row: row['row']), 1):
        case = Path(record['source'])
        if not case.resolve().is_relative_to(writer.project / 'outputs'):
            raise ValueError(f'case outside project outputs: {case}')
        prediction_path = case / 'coordinator_1800/prediction.json'
        evaluation_path = case / 'evaluation/result.json'
        for path, field in ((prediction_path, 'prediction_sha256'), (evaluation_path, 'evaluation_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest() != record[field]:
                raise ValueError(f'frozen result changed: {path}')
        fingerprint = add_inference_source(writer, case / 'source', sources)
        if record.get('inference_source_sha256') and fingerprint != record['inference_source_sha256']:
            raise ValueError(f'frozen inference source mismatch: {case}')
        if full and not fingerprint:
            raise ValueError(f'latest case has no frozen inference source: {case}')
        prefix = f'{label}/cases/{case.name}'
        files = [writer.store(path, prefix + '/' + path.relative_to(case).as_posix(),
                              original_path=path.relative_to(case).as_posix())
                 for path in case_files(case, full)]
        if full:
            names = {entry['path'] for entry in files}
            if not {'nodes_1800/calls.jsonl', 'coordinator_1800/calls.jsonl'} <= names:
                raise ValueError(f'model journals missing: {case}')
            for log in sorted((writer.project / 'outputs').glob(case.name.replace('bank_case_', 'bank_case-', 1) + '.*')):
                if log.is_file() and log.suffix in ('.log', '.err'):
                    files.append(writer.store(log, prefix + '/slurm_logs/' + log.name,
                                              original_path='slurm_logs/' + log.name))
        evaluation = load(evaluation_path)
        score = {}
        for tolerance in (60, 300):
            _, _, value = evaluate(prediction_path.read_text(), evaluation['scoring_points'],
                                   time_tolerance_seconds=tolerance)
            score[str(tolerance)] = value
        recorded_tolerance = str(evaluation.get('time_tolerance_seconds', 60))
        if recorded_tolerance in score and abs(score[recorded_tolerance] - evaluation['score']) > 1e-12:
            raise ValueError(f'archived evaluation does not match scoring: {case}')
        metadata = {'row': record['row'], 'original_case': case.relative_to(writer.project).as_posix(),
                    'inference_source_sha256': fingerprint, 'scope': 'full_generated_evidence' if full else 'final_results_only',
                    'raw_database_included': False, 'files': files}
        case_manifest_path = prefix + '/case_manifest.json'
        write_json(safe_path(writer.output, case_manifest_path), metadata)
        cases.append({'row': record['row'], 'manifest': case_manifest_path, 'scores': score,
                      'final_reply_valid': evaluation['final_reply_valid'], 'inference_source_sha256': fingerprint,
                      'file_count': len(files)})
        scores.append({'row': record['row'], 'original_case': metadata['original_case'], 'scores': score,
                       'final_reply_valid': evaluation['final_reply_valid'],
                       'prediction': load(prediction_path), 'instruction': evaluation['instruction'],
                       'inference_source_sha256': fingerprint})
        if number % 10 == 0 or number == expected:
            print(json.dumps({'run': label, 'completed': number, 'total': expected,
                              'stored_mib': round(writer.counts['stored_bytes'] / 1024**2, 1)}), flush=True)
    summaries = {str(tolerance): {'questions': expected,
                 'fully_correct_count': sum(row['scores'][str(tolerance)] == 1 for row in scores),
                 'any_field_correct_count': sum(row['scores'][str(tolerance)] > 0 for row in scores),
                 'mean_field_score': sum(row['scores'][str(tolerance)] for row in scores) / expected}
                 for tolerance in (60, 300)}
    write_json(writer.output / label / 'summary.json', summaries)
    (writer.output / label / 'results.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in scores), encoding='utf-8')
    return {'source_arrays': original['source_arrays'], 'questions': expected, 'cases': cases,
            'scope': 'all generated case JSON, calls and reports; no inbox database' if full else 'final predictions and evaluations only',
            'summary': summaries}


def export(project: Path, output: Path, threshold=65536, level=6):
    project, output = project.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f'archive output must be a new directory: {output}')
    output.mkdir(parents=True)
    writer, sources, extra = ArchiveWriter(project, output, threshold, level), {}, []
    runs = {
        'latest': archive_run(writer, project / 'outputs/staged_full_20261102/manifest.json',
                              'latest', sources, full=True),
        'previous_full': archive_run(writer, project / 'outputs/completion_20261102/original_full/manifest.json',
                                     'previous_full', sources, full=False),
    }
    reports = ['latest_experiment_review_20261103', 'completion_20261102',
               'recovery_verification_20261102', 'score_repair_20261101', 'repair_20261101',
               'rollback_20261030', 'trace_flow_redesign_20261103',
               'trace_counts_clarity_20261103', 'trace_handoff_clarity_20261103']
    for name in reports:
        for path in sorted((project / 'outputs' / name).iterdir()):
            allowed = TEXT_SUFFIXES - {'.err'}
            if name == 'rollback_20261030':
                allowed = {'.md', '.json'}
            if path.is_file() and not path.is_symlink() and path.suffix in allowed:
                extra.append(writer.store(path, f'reports/{name}/{path.name}'))
    for name in ['staged_full_20261102', 'completion_20261102/original_full']:
        for path in sorted((project / 'outputs' / name).iterdir()):
            if path.is_file() and path.suffix in ('.md', '.json'):
                extra.append(writer.store(path, f'reports/{name}/{path.name}'))
    references = {}
    for label, source in [
            ('rollback_20261030', 'outputs/rollback_20261030/frozen_source'),
            ('offline_candidate_20261102', 'outputs/offline_baseline_20261102/source'),
            ('accepted_retrieval_20261101', 'outputs/score_repair_20261101/retrieval_source')]:
        directory = project / source
        references[label] = {'inference_source_sha256': source_fingerprint(directory),
                             'files': copy_tree(writer, directory, 'reference_sources/' + label)}
        extra.extend(references[label]['files'])
    for name in ('manifest.json', 'plan.json', 'recovery_status.json'):
        path = project / 'outputs/offline_baseline_20261102' / name
        extra.append(writer.store(path, 'reports/offline_baseline_20261102/' + name))
    frontend = []
    for path in sorted((project / 'web/trace/data').glob('*.json')):
        frontend.append(writer.store(path, 'frontend/' + path.name, original_path=path.name))
    if len(frontend) != 4:
        raise ValueError('expected index.json and the three saved frontend case JSON files')
    extra.extend(frontend)
    for name in ('desktop_IG01_handoff.png', 'desktop_IG01_selected.png', 'desktop_IG01_routes.png'):
        path = project / 'outputs/trace_handoff_clarity_20261103' / name
        extra.append(writer.store(path, 'reports/trace_handoff_clarity_20261103/' + name))
    for source in sources.values():
        extra.extend(source['files'])
    stats = dict(writer.counts)
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'runs': runs, 'sources': sources, 'reference_sources': references,
                'frontend': {'rows': [1, 20, 128], 'files': frontend}, 'files': extra,
                'statistics': stats, 'compression': {'format': 'gzip', 'level': level,
                 'json_threshold_bytes': threshold, 'model_call_jsonl': 'original plain text'},
                'not_included': ['full telemetry datasets', 'frames/inbox.sqlite and other raw databases',
                                 'model weights and virtual environments', 'duplicate large backup archives',
                                 'full model transcripts from older runs other than latest'],
                'evaluation_policy': 'gold labels are evaluation-only; never append them to inference messages',
                'new_model_calls': 0, 'new_gpu_jobs': 0}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'completed': True, 'output': str(output), **stats,
                      'scores': runs['latest']['summary']}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('experiments/records'))
    parser.add_argument('--compression', type=int, choices=range(1, 10), default=6)
    args = parser.parse_args()
    export(args.project, args.output, level=args.compression)


if __name__ == '__main__':
    main()
