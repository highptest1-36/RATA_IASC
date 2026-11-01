"""Build the 27-class manifest and the T=64 skeleton/inertial arrays used in the paper from the original UTD-MHAD data.

Download Skeleton.zip and Inertial.zip from https://personal.utdallas.edu/~kehtar/UTD-MHAD.html, extract both into one
folder, and run from the repository root:

    python scripts/prepare_utd_mhad.py --extracted /path/to/extracted

Output (default): processed/UTD_MHAD_original_SI/{manifest_27class.csv, utd_mhad_skeleton_T64_f16.npy,
utd_mhad_inertial_T64_f16.npy, utd_mhad_preprocess_T64.json}
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src_utd_mhad_si.dataset import build_manifest, preprocess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--extracted', required=True, help='folder that contains the extracted Skeleton/ and Inertial/ .mat files')
ap.add_argument('--out', default=str(ROOT / 'processed/UTD_MHAD_original_SI'))
ap.add_argument('--overwrite', action='store_true')
args = ap.parse_args()

out = Path(args.out)
manifest = build_manifest(args.extracted, out / 'manifest_27class.csv')
preprocess(manifest, out, target_t=64, overwrite=args.overwrite)
