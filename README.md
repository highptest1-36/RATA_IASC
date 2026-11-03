# RATA-MAR: Reliability-Aware Skeleton–Inertial Fusion for Robust Human Action Recognition

Code, trained models, and per-seed results for the manuscript

> **Reliability-Aware Skeleton–Inertial Fusion for Robust Human Action Recognition under Missing and Degraded Modalities**
> Cao-Phuc Ha, Anh Nguyen Vu Minh, Khang Le Duy Dang, Phong Ho Huynh, and Phu-Nguyen Le.
> Submitted to *Intelligent Automation & Soft Computing* (under review).

RATA-MAR (Reliability-Aware Temporal Alignment for Multimodal Action Recognition) fuses a depth-camera skeleton stream
and a body-worn inertial (IMU) stream. A token-wise reliability estimator scores every temporal token before any
cross-modal exchange; the same scores gate a bidirectional cross-modal attention and set adaptive fusion weights. A
dual-view scheme trains one set of parameters on a clean and a synthetically corrupted view of every sample.

## Main results (UTD-MHAD, cross-subject, top-1 accuracy %, mean ± SD over three training seeds)

| Model | Clean | Missing skeleton | Missing IMU | Skeleton noise | IMU noise | Joint dropout 30% | IMU delay ±16 |
|---|---|---|---|---|---|---|---|
| LateFusion-Clean | 93.33 ± 0.71 | 3.72 ± 0.00 | 9.07 ± 1.76 | 80.43 ± 9.02 | 92.96 ± 0.93 | 75.92 ± 6.94 | 91.09 ± 1.07 |
| LateFusion-Robust | 93.57 ± 1.28 | 85.43 ± 0.88 | 58.91 ± 0.59 | 92.09 ± 1.08 | 93.53 ± 1.28 | 92.73 ± 0.50 | 92.13 ± 0.89 |
| **RATA-MAR** | **95.04 ± 0.82** | **87.91 ± 0.70** | **80.47 ± 1.01** | **93.91 ± 1.28** | **95.27 ± 1.21** | **93.29 ± 1.49** | **93.76 ± 1.22** |

The repository also contains the additional analyses of the revision: paired significance tests, 22 extended
degradation scenarios not seen during training, a fine delay sweep with an attention-offset analysis, degradation
detection and selective prediction, sensitivity to the token count and to the reliability targets, and a controlled
comparison with re-implemented TMC-style and QMF-style quality-aware fusion. All results, including the scenarios in
which RATA-MAR is less robust than other methods, are in `REVISION_R1/experiments/outputs/`.

## Repository layout

| Path | Content |
|---|---|
| `src_rata_mar_si_v21/` | Model (`models_v21.py`), training and evaluation protocol (`train_v21.py`), plotting |
| `src_utd_mhad_si/` | UTD-MHAD loading and preprocessing (27-class manifest, resampling to 64 time steps) |
| `scripts/prepare_utd_mhad.py` | Builds the processed arrays from the original UTD-MHAD files |
| `RATA_MAR_SI_v21_Publication_Safe.ipynb` | Full training pipeline of the paper (development split, final 3-seed training, ablation, efficiency), with outputs |
| `runs/RATA_MAR_SI_v21/tables/` | Per-seed and summary results of the paper |
| `REVISION_R1/experiments/` | Scripts of the additional analyses and models of the revision |
| `REVISION_R1/experiments/outputs/cuda/` | Results of the inference-only analyses (A100) |
| `REVISION_R1/experiments/outputs/gpu_train/` | Results of the additional training runs (token count, reliability targets, TMC-/QMF-style) |

## Setup

```bash
git clone https://github.com/highptest1-36/RATA_IASC.git
cd RATA_IASC
pip install -r requirements.txt     # PyTorch with CUDA: see https://pytorch.org
```

The results were produced with Python 3.13, PyTorch 2.11, and one NVIDIA A100 GPU.

## Data

UTD-MHAD is not redistributed here. Download `Skeleton.zip` and `Inertial.zip` from the
[official dataset page](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html), extract both into one folder, and run

```bash
python scripts/prepare_utd_mhad.py --extracted /path/to/extracted
```

This writes `processed/UTD_MHAD_original_SI/` (861 sequences, 27 actions, 8 subjects; skeleton `[861, 64, 20, 3]`,
inertial `[861, 64, 6]`, float16). The arrays are bit-identical to the ones used in the paper.

## Trained models

The checkpoints are attached to the [v1.0 release](https://github.com/highptest1-36/RATA_IASC/releases/tag/v1.0).
Unzip them in the repository root:

```bash
wget https://github.com/highptest1-36/RATA_IASC/releases/download/v1.0/checkpoints_published_v21.zip
wget https://github.com/highptest1-36/RATA_IASC/releases/download/v1.0/checkpoints_revision_r1.zip
unzip checkpoints_published_v21.zip && unzip checkpoints_revision_r1.zip
```

* `checkpoints_published_v21.zip` → `runs/RATA_MAR_SI_v21/final_models/`: the 21 models of the paper (RATA-MAR, the
  baselines, and the two ablation variants; three training seeds each, last-epoch weights).
* `checkpoints_revision_r1.zip` → `REVISION_R1/experiments/outputs/gpu_train/final_models/`: the 25 additional models
  of the revision (K ∈ {4, 16, 32}, three alternative reliability-target sets, TMC-style, QMF-style, and one
  re-training of the published configuration).

## Reproducing the results

Re-evaluate the published checkpoints and compare every per-seed number with the paper:

```bash
python REVISION_R1/experiments/r1_inference_analyses.py --stages A --force
# -> REVISION_R1/experiments/outputs/<device>/A_reproduction_check.csv (column abs_diff_pp)
```

On an A100 GPU, all 135 per-seed values reproduce exactly. On a CPU, floating-point and random-number differences
change individual values by up to about one percentage point.

Additional analyses of the revision (inference only, published checkpoints):

```bash
python REVISION_R1/experiments/r1_inference_analyses.py --stages ABCDE   # statistics, extended tests, delay sweep, detection
python REVISION_R1/experiments/r1_oracle_gate.py                         # fixed-reliability diagnostic
python REVISION_R1/experiments/r1_gap_diagnostic.py                      # zero-filled gaps vs. last-frame filling
python REVISION_R1/experiments/r1_baseline_tests.py                      # paired tests vs. TMC-/QMF-style (needs revision checkpoints)
```

Additional training (A100, about one minute per run; resumable):

```bash
python REVISION_R1/experiments/r1_train_variants_gpu.py --groups R0,K,T,B
```

The original models are trained with `RATA_MAR_SI_v21_Publication_Safe.ipynb` (Colab; set `PROJECT` in Section 2 to the
repository root). The official cross-subject protocol trains on subjects 1, 3, 5, and 7 and tests on subjects 2, 4,
6, and 8 with a fixed schedule of 100 epochs and the last-epoch weights; development decisions used subject 7 as
validation only. Scripts locate the repository root automatically; set `RATA_PROJECT` to override it.

## Contact

Corresponding author: Phu-Nguyen Le (nguyenlp@vaa.edu.vn), Vietnam Aerospace University.

First author: Cao-Phuc Ha (caophucai@gmail.com), FPT University.
