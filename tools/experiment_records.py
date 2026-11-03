"""Read and verify archived experiment evidence without models or datasets."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'experiments/records'


def safe_path(root: Path, name: str) -> Path:
    part = PurePosixPath(name)
    if not name or part.is_absolute() or '..' in part.parts or '\\' in name:
        raise ValueError(f'unsafe archive path: {name!r}')
    target = root.joinpath(*part.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'archive path escapes root: {name!r}')
    return target


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def case_manifest(root: Path, row: int, run='latest') -> tuple[dict, dict]:
    manifest = read_json(root / 'manifest.json')
    match = next((case for case in manifest['runs'][run]['cases'] if case['row'] == row), None)
    if match is None:
        raise ValueError(f'row {row} not archived in run {run}')
    return manifest, read_json(safe_path(root, match['manifest']))


@contextmanager
def record_stream(root: Path, entry: dict):
    path = safe_path(root, entry['stored_path'])
    if entry['compression'] == 'gzip':
        with gzip.open(path, 'rb') as stream:
            yield stream
    elif entry['compression'] == 'none':
        with path.open('rb') as stream:
            yield stream
    else:
        raise ValueError(f"unknown compression: {entry['compression']}")


def verify_entry(root: Path, entry: dict, destination: Path | None = None) -> int:
    digest, total = hashlib.sha256(), 0
    output = None
    try:
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = destination.open('xb')
        with record_stream(root, entry) as stream:
            while chunk := stream.read(128 * 1024):
                total += len(chunk)
                if total > entry['raw_bytes']:
                    raise ValueError(f"record exceeds declared size: {entry['stored_path']}")
                digest.update(chunk)
                if output:
                    output.write(chunk)
        if total != entry['raw_bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError(f"record checksum mismatch: {entry['stored_path']}")
    finally:
        if output:
            output.close()
    return total


def all_entries(root: Path):
    manifest = read_json(root / 'manifest.json')
    yield from manifest['files']
    for run in manifest['runs'].values():
        for case in run['cases']:
            yield from read_json(safe_path(root, case['manifest']))['files']


def materialize(root: Path, row: int, output: Path, run='latest') -> int:
    manifest, case = case_manifest(root, row, run)
    if output.exists():
        raise FileExistsError(f'output must be a new directory: {output}')
    # Validate every destination before creating files.
    for entry in case['files']:
        safe_path(output, entry['path'])
    output.mkdir(parents=True)
    for entry in case['files']:
        verify_entry(root, entry, safe_path(output, entry['path']))
    source = manifest.get('sources', {}).get(case.get('inference_source_sha256'))
    if source:
        for entry in source['files']:
            target = safe_path(output / 'source', entry['path'])
            verify_entry(root, entry, target)
    return len(case['files'])


def restore_frontend(root: Path, output: Path) -> int:
    manifest = read_json(root / 'manifest.json')
    entries = manifest['frontend']['files']
    for entry in entries:
        safe_path(output, entry['path'])
    if output.exists():
        raise FileExistsError(f'output must be a new directory: {output}')
    output.mkdir(parents=True)
    for entry in entries:
        verify_entry(root, entry, safe_path(output, entry['path']))
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    ls = commands.add_parser('list', help='list original record paths for a question')
    ls.add_argument('--row', type=int, required=True)
    ls.add_argument('--run', choices=['latest', 'previous_full'], default='latest')
    ls.add_argument('--contains', default='')
    show = commands.add_parser('show', help='print an original UTF-8 record, decompressing if needed')
    show.add_argument('--row', type=int, required=True)
    show.add_argument('--run', choices=['latest', 'previous_full'], default='latest')
    show.add_argument('--path', required=True)
    show.add_argument('--lines', type=int, help='print only the first N original lines')
    check = commands.add_parser('verify', help='verify original-byte SHA256 of archived evidence')
    check.add_argument('--row', type=int)
    check.add_argument('--run', choices=['latest', 'previous_full'], default='latest')
    extract = commands.add_parser('materialize', help='restore a question into a NEW directory')
    extract.add_argument('--row', type=int, required=True)
    extract.add_argument('--run', choices=['latest', 'previous_full'], default='latest')
    extract.add_argument('--output', type=Path, required=True)
    frontend = commands.add_parser('frontend', help='restore the three saved frontend cases')
    frontend.add_argument('--output', type=Path, default=Path('web/trace/data'))
    args = parser.parse_args()
    if args.command == 'list':
        _, case = case_manifest(args.root, args.row, args.run)
        for entry in case['files']:
            if args.contains in entry['path']:
                print(f"{entry['path']}\t{entry['raw_bytes']} bytes\t{entry['compression']}")
    elif args.command == 'show':
        _, case = case_manifest(args.root, args.row, args.run)
        entry = next((entry for entry in case['files'] if entry['path'] == args.path), None)
        if entry is None:
            raise ValueError(f'path not in question manifest: {args.path}')
        import sys
        with record_stream(args.root, entry) as stream:
            if args.lines is None:
                shutil.copyfileobj(stream, sys.stdout.buffer)
            else:
                if args.lines < 0:
                    raise ValueError('--lines must be nonnegative')
                for _ in range(args.lines):
                    line = stream.readline()
                    if not line:
                        break
                    sys.stdout.buffer.write(line)
    elif args.command == 'verify':
        entries = (case_manifest(args.root, args.row, args.run)[1]['files']
                   if args.row is not None else all_entries(args.root))
        count = total = 0
        for entry in entries:
            total += verify_entry(args.root, entry)
            count += 1
        print(json.dumps({'verified_files': count, 'original_bytes': total}))
    elif args.command == 'materialize':
        count = materialize(args.root, args.row, args.output, args.run)
        print(json.dumps({'restored_records': count, 'output': str(args.output),
                          'raw_database_included': False, 'models_called': False}))
    else:
        count = restore_frontend(args.root, args.output)
        print(json.dumps({'restored_frontend_files': count, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
