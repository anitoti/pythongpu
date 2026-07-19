from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import hashlib
from datetime import datetime
import subprocess


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Register a DTI .mat into data/ with standardized name')
    parser.add_argument('--src', required=True, help='source .mat file')
    parser.add_argument('--tag', required=True, help='short tag for filename')
    parser.add_argument('--move', action='store_true', help='move instead of copy')
    parser.add_argument('--outdir', default='data', help='destination directory')
    parser.add_argument('--compute', action='store_true', help='run spectral diagnostics after registering')
    parser.add_argument('--force', action='store_true', help='force recompute')
    parser.add_argument('--index-file', default='dti_spectra_index.json', help='index JSON file')
    args = parser.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f'source not found: {src}')
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target = outdir / f'DTI-{args.tag}.mat'
    if target.exists():
        # compare hashes
        with open(src, 'rb') as f: src_hash = hashlib.sha256(f.read()).hexdigest()
        with open(target, 'rb') as f: tgt_hash = hashlib.sha256(f.read()).hexdigest()
        if src_hash == tgt_hash:
            print(f'source already present at {target} (identical).')
        else:
            suffix = datetime.now().strftime('%Y%m%dT%H%M%S')
            target = outdir / f'DTI-{args.tag}-{suffix}.mat'

    if args.move:
        shutil.move(str(src), str(target))
    else:
        shutil.copy2(str(src), str(target))

    print(f'Registered DTI file -> {target}')

    if args.compute:
        cmd = [ 'python3', 'scripts/run_dti_spectral.py', '--mat', str(target), '--index-file', args.index_file ]
        if args.force:
            cmd.append('--force')
        print('Running diagnostics:', ' '.join(cmd))
        subprocess.check_call(cmd)


if __name__ == '__main__':
    main()
