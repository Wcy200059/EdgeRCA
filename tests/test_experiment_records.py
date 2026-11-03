"""Published evidence must preserve original bytes and restore into new paths."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.experiment_records import materialize, restore_frontend, safe_path, verify_entry
from tools.export_experiment_records import ArchiveWriter, case_files


class ExperimentRecordTests(unittest.TestCase):
    def test_compression_preserves_original_bytes_and_deterministic_blobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary); output = project / 'archive'
            source = project / 'original.json'
            raw = json.dumps({'messages': [{'role': 'user', 'content': '原样证据'}]}, ensure_ascii=False).encode() + b'\n'
            source.write_bytes(raw)
            writer = ArchiveWriter(project, output, threshold=1)
            first = writer.store(source, 'one.json', original_path='original.json')
            second = writer.store(source, 'two.json', original_path='original.json')
            self.assertEqual(first['stored_sha256'], second['stored_sha256'])
            self.assertEqual(gzip.decompress((output / first['stored_path']).read_bytes()), raw)
            self.assertEqual(first['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(verify_entry(output, first), len(raw))
            first['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_entry(output, first)

    def test_raw_databases_models_and_source_copies_are_not_case_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            for name in ['frames/inbox.sqlite', 'prepared/records.jsonl', 'model.safetensors',
                         'source/edge_rca/a.py', 'nodes_1800/calls.jsonl',
                         'coordinator_1800/queries/q1/packet.json', 'frames/c1800/device_A.json']:
                path = case / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
            paths = {p.relative_to(case).as_posix() for p in case_files(case, full=True)}
            self.assertEqual(paths, {'nodes_1800/calls.jsonl', 'coordinator_1800/queries/q1/packet.json',
                                     'frames/c1800/device_A.json'})

    def test_materialize_and_frontend_restore_do_not_overwrite_existing_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary); output = project / 'archive'
            source = project / 'calls.jsonl'; source.write_text('{"raw_reply":"kept"}\n')
            entry = ArchiveWriter(project, output).store(source, 'case/calls.jsonl', original_path='nodes_1800/calls.jsonl')
            (output / 'case/manifest.json').write_text(json.dumps({'files': [entry], 'inference_source_sha256': None}))
            (output / 'manifest.json').write_text(json.dumps({'runs': {'latest': {'cases': [
                {'row': 1, 'manifest': 'case/manifest.json'}]}}, 'sources': {}, 'frontend': {'files': [entry]}}))
            target = project / 'restored'
            self.assertEqual(materialize(output, 1, target), 1)
            self.assertEqual((target / entry['path']).read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError):
                materialize(output, 1, target)
            with self.assertRaises(FileExistsError):
                restore_frontend(output, target)

    def test_archive_paths_and_credentials_are_checked_before_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            for name in ('../other', '/absolute', 'sub/../../other', 'sub\\other'):
                with self.assertRaises(ValueError):
                    safe_path(project, name)
            source = project / 'record.json'
            source.write_text('-----BEGIN OPENSSH ' + 'PRIVATE KEY-----\n')
            with self.assertRaisesRegex(ValueError, 'possible credential'):
                ArchiveWriter(project, project / 'archive').store(source, 'record.json')


if __name__ == '__main__':
    unittest.main()
