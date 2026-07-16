#!/usr/bin/env python3
"""
Register a DTI .mat file into data/ with standardized filename and optionally
run spectral diagnostics to index it.

Naming convention: data/DTI-<tag>.mat
Examples:
    scripts/register_dti.py --src /tmp/DTI_A.mat --tag og --move --compute
    scripts/register_dti.py --src /path/sub-001_DTI.mat --tag sub-001 --compute
"""
from pathlib import Path
import argparse
import shutil
import hashlib
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--src', required=True, help='source .mat file path')
parser.add_argument('--tag', required=True, help='short tag to embed in filename (e.g., og, sub-01)')
parser.add_argument('--move', action='store_true', help='move instead of copy')
parser.add_argument('--outdir', default='data', help='directory to place standardized file')
parser.add_argument('--compute', action='store_true', help='run spectral diagnostics after registering')
parser.add_argument('--force', action='store_true', help='pass --force to diagnostics to recompute if indexed')
parser.add_argument('--index-file', default='dti_spectra_index.json', help='index JSON file to pass through')
args = parser.parse_args()

src = Path(args.src)
if not src.exists():
    raise SystemExit(f'source not found: {src}')

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# target filename
base = f'DTI-{args.tag}.mat'
target = outdir / base
if target.exists():
    # if identical file, keep; otherwise make unique with timestamp
    with open(src, 'rb') as f: src_hash = hashlib.sha256(f.read()).hexdigest()
    with open(target, 'rb') as f: tgt_hash = hashlib.sha256(f.read()).hexdigest()
    if src_hash == tgt_hash:
        print(f'source already present at {target} (identical).')
    else:
        from datetime import datetime
        suffix = datetime.now().strftime('%Y%m%dT%H%M%S')
        target = outdir / f'DTI-{args.tag}-{suffix}.mat'

# copy or move
if args.move:
    shutil.move(str(src), str(target))
else:
    shutil.copy2(str(src), str(target))

print(f'Registered DTI file -> {target}')

# Optionally run diagnostics
if args.compute:
    cmd = [ 'python3', 'scripts/run_dti_spectral.py', '--mat', str(target), '--index-file', args.index_file ]
    if args.force:
        cmd.append('--force')
    print('Running diagnostics:', ' '.join(cmd))
    subprocess.check_call(cmd)

print('Done.')
