"""Verify restored sources and report the offline Bank repeat without model reruns."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted((Path(root) / 'edge_rca').glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rescore(case):
    from edge_rca.scoring import evaluate
    path = Path(case) / 'evaluation'
    report = json.loads((path / 'result.json').read_text())
    for tolerance in (60, 300):
        passed, failed, score = evaluate(json.dumps(report['prediction']), report['scoring_points'], time_tolerance_seconds=tolerance)
        result = deepcopy(report)
        result.update(score=score, passed=passed, failed=failed,
                      fully_correct=score == 1, any_field_correct=score > 0,
                      time_tolerance_seconds=tolerance,
                      scoring_rule=f'components/reasons exact; occurrence time <= {tolerance}s')
        output = path / f'result_{tolerance}s.json'
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'row': report['row'], 'final_reply_valid': report['final_reply_valid'], 'scored_at_seconds': [60, 300]}))


def summarize(root, expected):
    root = Path(root)
    output = {'expected_questions': expected, 'time_tolerances': {}}
    lines = ['# 离线回退复测结果', '', '同一份冻结源码，完整半小时回放后由4B节点与9B协调者完成诊断；保存答案后独立评分。', '',
             '| 时间容差 | 已完成题数 | 完全正确 | 至少一项正确（含完全正确） |', '|---|---:|---:|---:|']
    for tolerance in (60, 300):
        records = [json.loads(p.read_text()) for p in sorted(root.glob(f'cases/*/evaluation/result_{tolerance}s.json'))]
        count = len(records)
        full = sum(r['fully_correct'] for r in records)
        partial = sum(r['any_field_correct'] for r in records)
        rows = {r['row'] for r in records}
        if len(rows) != count:
            raise ValueError('duplicate question rows')
        output['time_tolerances'][str(tolerance)] = {'completed': count, 'fully_correct': full, 'any_field_correct': partial,
            'invalid_replies': [r['row'] for r in records if not r['final_reply_valid']],
            'missing_rows': sorted(set(range(expected)) - rows),
            'mean_field_score': sum(r['score'] for r in records) / count if count else None}
        full_text = f'{full/count:.2%}（{full}/{count}）' if count else '尚未完成'
        partial_text = f'{partial/count:.2%}（{partial}/{count}）' if count else '尚未完成'
        lines.append(f'| ±{tolerance}秒 | {count}/{expected} | {full_text} | {partial_text} |')
    output['complete'] = all(v['completed'] == expected for v in output['time_tolerances'].values())
    lines += ['', '历史144524全136题：官方±60秒完全正确11.76%（16/136）、至少一项正确31.62%（43/136）；±300秒分别为16.18%（22/136）、39.71%（54/136）。',
              '当前仅完成部分题目时，百分比只以已完成题目为分母，不能与历史全136题直接比较。',
              '旧源码目录已删除，恢复代码由历史记录重建，指纹尚未与旧记录完全一致；本次用于验证恢复后的实际效果。', '']
    (root / 'summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    (root / '复测结果.md').write_text('\n'.join(lines))
    print(json.dumps(output, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--source', type=Path, required=True)
    verify.add_argument('--manifest', type=Path, required=True)
    single = sub.add_parser('rescore')
    single.add_argument('--case', type=Path, required=True)
    summary = sub.add_parser('summary')
    summary.add_argument('--root', type=Path, required=True)
    summary.add_argument('--expected', type=int, default=136)
    args = parser.parse_args()
    if args.action == 'verify':
        manifest = json.loads(args.manifest.read_text())
        actual = source_hash(args.source)
        if actual != manifest['source_sha256']:
            raise ValueError('source differs from frozen restore manifest')
        print('Verified source: ' + actual)
    elif args.action == 'rescore':
        rescore(args.case)
    else:
        summarize(args.root, args.expected)


if __name__ == '__main__':
    main()
