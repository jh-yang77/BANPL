#!/usr/bin/env python3
"""Collect legacy BANPL in/out scores or NPZ arrays using an explicit run manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from protocols import GROUPS, DATASETS, COUNTS


def measures(id_scores, ood_scores, direction='id_high'):
    """ID is positive; nearest recall convention matches the historical BANPL evaluator."""
    a, b = np.asarray(id_scores, dtype=float), np.asarray(ood_scores, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or not a.size or not b.size:
        raise ValueError('Both score arrays must be nonempty and one-dimensional')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite score')
    if direction not in ('id_high', 'ood_high'):
        raise ValueError('Specify score direction: id_high or ood_high')
    y = np.r_[np.ones(a.size), np.zeros(b.size)]
    s = np.r_[a, b] * (1 if direction == 'id_high' else -1)
    order = np.argsort(s, kind='mergesort')[::-1]
    labels, scores = y[order], s[order]
    ends = np.r_[np.where(np.diff(scores))[0], len(scores)-1]
    tp = np.cumsum(labels, dtype=np.float64)[ends]
    fp = 1 + ends - tp
    recall = tp / tp[-1]
    sl = slice(tp.searchsorted(tp[-1]), None, -1)
    recalls = np.r_[recall[sl], 1]
    fps = np.r_[fp[sl], 0]
    return dict(fpr95=100 * fps[np.argmin(abs(recalls-.95))] / b.size,
                auroc=100 * roc_auc_score(y, s),
                aupr_id=100 * average_precision_score(y, s))


def read_scores(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as d:
            return d['id_scores'], d['ood_scores']
    groups = {'in': [], 'out': []}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            label, score = line.split()
            if label not in groups:
                raise ValueError(f'{path}: unknown label {label!r}')
            groups[label].append(float(score))
    return np.array(groups['in']), np.array(groups['out'])


def aggregate(rows):
    """Average datasets within each run first; then mean and sample SD across runs."""
    if not rows:
        raise ValueError('No measurements')
    keys = [(r['method'], r['backbone'], r['seed'], r['dataset']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate method/backbone/seed/dataset')
    expanded = list(rows)
    runs = sorted({(r['method'], r['backbone'], r['seed']) for r in rows})
    for method, backbone, seed in runs:
        rr = {r['dataset']: r for r in rows if (r['method'], r['backbone'], r['seed']) == (method, backbone, seed)}
        if set(rr) != set(DATASETS):
            raise ValueError(f'{method}/{backbone}/{seed}: require all seven datasets; got {sorted(rr)}')
        for group, datasets in GROUPS.items():
            expanded.append(dict(method=method, backbone=backbone, seed=seed, dataset=group,
                **{k: float(np.mean([rr[d][k] for d in datasets])) for k in ('fpr95','auroc','aupr_id')}))
    means = []
    for method, backbone, dataset in sorted({(r['method'],r['backbone'],r['dataset']) for r in expanded}):
        rr = [r for r in expanded if (r['method'],r['backbone'],r['dataset']) == (method,backbone,dataset)]
        result = dict(method=method, backbone=backbone, dataset=dataset, runs=len(rr))
        for key in ('fpr95','auroc','aupr_id'):
            values = [r[key] for r in rr]
            result[key] = float(np.mean(values))
            result[key+'_std'] = float(np.std(values, ddof=1)) if len(values)>1 else ''
        means.append(result)
    return expanded, means


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True, type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('id_dataset') != 'ImageNet-1K':
        raise ValueError('Only ImageNet-1K is supported')
    rows=[]; evidence=[]
    for run in manifest['runs']:
        if set(run['scores']) != set(DATASETS):
            raise ValueError('Each run must provide exactly the seven OOD datasets')
        for dataset, filename in run['scores'].items():
            path=Path(filename)
            if not path.is_absolute(): path=args.manifest.parent/path
            a,b=read_scores(path)
            if len(a)!=50000 or len(b)!=COUNTS[dataset]:
                raise ValueError(f'{dataset}: expected 50000/{COUNTS[dataset]} ID/OOD samples; got {len(a)}/{len(b)}')
            rows.append(dict(method=run['method'],backbone=run['backbone'],seed=str(run['seed']),dataset=dataset,
                **measures(a,b,run['direction'])))
            evidence.append(dict(file=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                method=run['method'],backbone=run['backbone'],seed=run['seed'],dataset=dataset,direction=run['direction']))
    expanded, means=aggregate(rows)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/'per_run.csv',expanded)
    write_csv(args.output_dir/'mean_results.csv',means)
    (args.output_dir/'provenance.local.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(f'Collected {len(rows)} measurements; units: percent; FPR95: nearest ID recall')

if __name__=='__main__':main()
