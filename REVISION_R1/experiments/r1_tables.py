"""Generate the LaTeX tables of Revision 1 from the result CSVs (manuscript_R1/generated/*.tex).

Published numbers (Tables 2, 3, 6) come from runs/RATA_MAR_SI_v21/tables and are never recomputed.
New analyses come from experiments/outputs/<device>; 'cuda' is preferred, 'cpu' is used as a draft and is
flagged with a PENDING note so that the final submission uses the A100 run.

Usage: python -I r1_tables.py
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get('RATA_PROJECT', Path(__file__).resolve().parents[2]))  # repository root
TABLES = PROJECT / 'runs/RATA_MAR_SI_v21/tables'
EXP = PROJECT / 'REVISION_R1/experiments'
OUT = EXP / 'outputs'
GEN = PROJECT / 'REVISION_R1/manuscript_R1/generated'
GEN.mkdir(parents=True, exist_ok=True)
NAMES = json.loads((OUT / 'ext_names.json').read_text())

SRC = OUT / 'cuda' if (OUT / 'cuda' / 'B_paired_tests.csv').exists() else OUT / 'cpu'
DRAFT = SRC.name == 'cpu'
SCEN_LABEL = {'clean': 'Clean', 'missing_skeleton': 'Missing skeleton', 'missing_imu': 'Missing IMU',
              'skeleton_noise': 'Skeleton noise', 'imu_noise': 'IMU noise', 'joint_drop_30': 'Joint dropout (30\\%)',
              'delay_pm4': 'Delay $\\pm 4$', 'delay_pm8': 'Delay $\\pm 8$', 'delay_pm16': 'Delay $\\pm 16$'}


def ms(v, bold=False, nd=2):
    v = np.asarray(v, dtype=float) * 100
    s = f'{v.mean():.{nd}f} $\\pm$ {v.std(ddof=1):.{nd}f}'
    return f'\\textbf{{{s}}}' if bold else s


def write(name, text):
    (GEN / name).write_text(text.strip() + '\n', encoding='utf-8')
    print('wrote', name)


def draft_note():
    return ' \\PENDING{draft from the CPU re-evaluation; replace by the A100 run}' if DRAFT else ''


# ---------------------------------------------------------------- published tables with SD
def table_clean():
    d = pd.read_csv(TABLES / 'FINAL_XSUB_per_seed.csv')
    rows = [('SkeletonOnly', 'No'), ('IMUOnly', 'No'), ('LateFusion-Clean', 'No'), ('LateFusion-Robust', 'Yes'),
            ('RATA-MAR-SI-v2.1', 'Yes')]
    lines = []
    for m, rob in rows:
        g = d[d.model == m]
        best = m == 'RATA-MAR-SI-v2.1'
        name = 'RATA-MAR (proposed method)' if best else m
        lines.append(f'{name} & {rob} & {ms(g.clean_top1, best)} & {ms(g.clean_macro_f1, best)}\\\\')
    write('table_clean_rows.tex', '\n'.join(lines))


def table_robust():
    d = pd.read_csv(TABLES / 'FINAL_XSUB_per_seed.csv')
    models = ['LateFusion-Clean', 'LateFusion-Robust', 'RATA-MAR-SI-v2.1']
    cols = ['clean_top1', 'missing_skeleton', 'missing_imu', 'skeleton_noise', 'imu_noise', 'joint_drop_30',
            'delay_pm4', 'delay_pm8', 'delay_pm16']
    lines = []
    for c in cols:
        means = {m: d[d.model == m][c].mean() for m in models}
        best = max(means, key=means.get)
        cells = [ms(d[d.model == m][c], m == best) for m in models]
        lines.append(f'{SCEN_LABEL["clean" if c == "clean_top1" else c]} & ' + ' & '.join(cells) + '\\\\')
    write('table_robust_rows.tex', '\n'.join(lines))


def table_ablation():
    d = pd.read_csv(TABLES / 'FINAL_Ablation_per_seed.csv')
    cols = ['clean_top1', 'missing_skeleton', 'missing_imu', 'joint_drop_30', 'imu_noise', 'delay_pm4', 'delay_pm8',
            'delay_pm16']
    st = pd.read_csv(SRC / 'B_paired_tests.csv')
    lines = []
    for c in cols:
        sc = 'clean' if c == 'clean_top1' else c
        means = {v: d[d.variant == v][c].mean() for v in ['Full', 'NoAlignment', 'NoReliability']}
        best = max(means, key=means.get)
        cells = []
        for v in ['Full', 'NoAlignment', 'NoReliability']:
            cell = ms(d[d.variant == v][c], v == best)
            if v != 'Full':
                r = st[(st.model_b == v) & (st.scenario == sc)]
                if len(r) and r.p_holm.iloc[0] < 0.05:
                    cell += '$^{\\ast}$'
            cells.append(cell)
        lines.append(f'{SCEN_LABEL[sc]} & ' + ' & '.join(cells) + '\\\\')
    write('table_ablation_rows.tex', '\n'.join(lines))


# ---------------------------------------------------------------- new analyses
def table_stats():
    st = pd.read_csv(SRC / 'B_paired_tests.csv')
    lines = []
    for b in ['LateFusion-Robust', 'NoAlignment', 'NoReliability']:
        g = st[st.model_b == b]
        lines.append(f'\\multicolumn{{5}}{{l}}{{\\textit{{RATA-MAR vs. {b}}}}}\\\\')
        for _, r in g.iterrows():
            p = r.p_holm
            ps = '$<$0.001' if p < 0.001 else f'{p:.3f}'
            sig = '$^{\\ast}$' if p < 0.05 else ''
            lines.append(f'\\quad {SCEN_LABEL[r.scenario]} & {r.delta_pp:+.2f}{sig} & '
                         f'[{r.ci_lo_pp:+.2f}, {r.ci_hi_pp:+.2f}] & {ps} & {int(r.seed_wins_a)}/3\\\\')
        lines.append('\\midrule')
    write('table_stats_rows.tex', '\n'.join(lines[:-1]))
    write('table_stats_note.tex', draft_note())


def table_extended():
    p = SRC / 'C_extended_per_run.csv'
    if not p.exists():
        return
    d = pd.read_csv(p)
    per_seed = d.groupby(['model', 'seed', 'scenario']).agg(top1=('top1', 'mean'), r_s=('r_skel', 'mean'),
                                                           r_i=('r_imu', 'mean'), a_s=('alpha_skel', 'mean')).reset_index()
    models = ['LateFusion-Clean', 'LateFusion-Robust', 'RATA-MAR']
    order = list(NAMES)
    lines, prev = [], None
    group = lambda s: 'Conflict' if 'conflict' in s else ('Both' if s.startswith(('both', 'jointdrop', 'skelnoise'))
                                                         else ('Skeleton' if s.startswith('skel') else 'Inertial'))
    for sc in order:
        g = per_seed[per_seed.scenario == sc]
        if g.empty:
            continue
        if prev is not None and group(sc) != prev:
            lines.append('\\midrule')
        prev = group(sc)
        means = {m: g[g.model == m].top1.mean() for m in models}
        best = max(means, key=means.get)
        cells = [ms(g[g.model == m].top1, m == best, nd=1) for m in models]
        r = g[g.model == 'RATA-MAR']
        lines.append(f'{NAMES[sc]} & ' + ' & '.join(cells) +
                     f' & {r.r_s.mean():.2f} & {r.r_i.mean():.2f} & {r.a_s.mean():.2f}\\\\')
    # summary row: mean over all extended scenarios
    lines.append('\\midrule')
    mean_rows = per_seed.groupby(['model', 'seed']).top1.mean().reset_index()
    cells = [ms(mean_rows[mean_rows.model == m].top1, m == 'RATA-MAR', nd=1) for m in models]
    lines.append('Mean of 22 scenarios & ' + ' & '.join(cells) + ' & -- & -- & --\\\\')
    write('table_extended_rows.tex', '\n'.join(lines))
    write('table_extended_note.tex', draft_note())
    # summary for the text
    summ = {}
    for m in ['LateFusion-Clean', 'LateFusion-Robust', 'RATA-MAR', 'NoAlignment', 'NoReliability']:
        mr = mean_rows[mean_rows.model == m].top1
        summ[m] = (mr.mean() * 100, mr.std(ddof=1) * 100)
    (OUT / f'summary_extended_{SRC.name}.json').write_text(json.dumps(summ, indent=1))


def table_detect():
    p = SRC / 'E_detection_auroc.csv'
    if not p.exists():
        return
    d = pd.read_csv(p)
    g = d.groupby(['scenario', 'score']).auroc.agg(['mean', 'std']).reset_index()
    order = ['missing_skeleton', 'missing_imu', 'skel_tracking_loss', 'skel_arm_occlusion', 'imu_intermittent',
             'imu_frozen', 'imu_bias', 'imu_drift', 'imu_gain', 'both_noise_strong', 'both_severe', 'imu_conflict',
             'skel_conflict']
    label = dict(NAMES, missing_skeleton='Missing skeleton', missing_imu='Missing IMU')
    lines = []
    for sc in order:
        row = [label[sc]]
        for s in ['low_reliability', 'aux_disagreement', 'low_confidence']:
            r = g[(g.scenario == sc) & (g.score == s)]
            row.append(f'{r["mean"].iloc[0]:.2f}')
        lines.append(' & '.join(row) + '\\\\')
    write('table_detect_rows.tex', '\n'.join(lines))


def table_variants():
    p = OUT / 'gpu_train' / 'variants_per_seed.csv'
    pub = pd.read_csv(TABLES / 'FINAL_XSUB_per_seed.csv')
    full = pub[pub.model == 'RATA-MAR-SI-v2.1']
    cols = ['clean_top1', 'missing_skeleton', 'missing_imu', 'skeleton_noise', 'joint_drop_30', 'delay_pm16']
    if not p.exists():
        write('table_variants_rows.tex', '\\multicolumn{7}{c}{\\PENDING{A100 training runs (K and target variants)}}\\\\')
        write('table_baselines_rows.tex', '\\multicolumn{8}{c}{\\PENDING{A100 training runs (TMC-/QMF-style)}}\\\\')
        return
    d = pd.read_csv(p)
    ext = SRC / 'C_extended_per_run.csv'
    ext_runs = pd.read_csv(SRC / 'C_extended_per_run.csv')
    ext_pub = ext_runs[ext_runs.model == 'RATA-MAR'].groupby(['seed', 'scenario']).top1.mean().groupby('seed').mean()

    def ext_cell(g, v):
        return ms(ext_pub, nd=1) if v is None else ms(g.extended_mean_top1, nd=1)
    lines = ['\\multicolumn{8}{l}{\\textit{Token count $K$ (time steps per token)}}\\\\']
    spec = [('K4', '$K = 4$ (16)'), (None, '$K = 8$ (8), paper'), ('K16', '$K = 16$ (4)'), ('K32', '$K = 32$ (2)')]
    for v, lab in spec:
        g = full if v is None else d[d.variant == v]
        lines.append(lab + ' & ' + ' & '.join(ms(g[c], nd=1) for c in cols) + ' & ' + ext_cell(g, v) + '\\\\')
    lines.append('\\midrule')
    lines.append('\\multicolumn{8}{l}{\\textit{Reliability targets (removal, dropout, noise)}}\\\\')
    spec = [(None, '(0, 0.60, 0.65), paper'), ('T_low', '(0, 0.40, 0.45)'), ('T_high', '(0, 0.80, 0.85)'),
            ('T_flat', '(0, 0.50, 0.50)')]
    for v, lab in spec:
        g = full if v is None else d[d.variant == v]
        lines.append(lab + ' & ' + ' & '.join(ms(g[c], nd=1) for c in cols) + ' & ' + ext_cell(g, v) + '\\\\')
    r0 = d[d.variant == 'R0_RATA_K8_default']
    write('table_variants_rows.tex', '\n'.join(lines))
    if len(r0):
        (OUT / 'gpu_train' / 'R0_reproduction.txt').write_text(
            f"retrained seed 0 clean {r0.clean_top1.iloc[0]*100:.2f} vs published {full[full.training_seed==0].clean_top1.iloc[0]*100:.2f}\n")
    # baselines with extended mean
    cols_b = ['clean_top1', 'missing_skeleton', 'missing_imu', 'skeleton_noise', 'joint_drop_30', 'delay_pm16']
    lines = []
    pub_lr = pub[pub.model == 'LateFusion-Robust']
    ext_df = pd.read_csv(ext) if ext.exists() else None

    def ext_mean(model):
        if ext_df is None:
            return '--'
        e = ext_df[ext_df.model == model].groupby(['seed', 'scenario']).top1.mean().groupby('seed').mean()
        return ms(e, nd=1)
    nr = pd.read_csv(TABLES / 'FINAL_Ablation_per_seed.csv').query('variant=="NoReliability"').copy()
    if 'skeleton_noise' not in nr.columns:   # not in the published ablation table; take the re-evaluation
        a_runs = pd.read_csv(SRC / 'A_original_scenarios_per_run.csv')
        sn = a_runs[(a_runs.model == 'NoReliability') & (a_runs.scenario == 'skeleton_noise')].groupby('seed').top1.mean()
        nr['skeleton_noise'] = sn.sort_index().values
    entries = [('LateFusion-Robust', pub_lr, ext_mean('LateFusion-Robust')),
               ('NoReliability (MulT-style)', nr, ext_mean('NoReliability'))]
    for v in ['TMC-style', 'QMF-style']:
        g = d[d.variant == v]
        entries.append((v, g, ms(g.extended_mean_top1, nd=1)))
    entries.append(('RATA-MAR (proposed method)', full, ext_mean('RATA-MAR')))
    # significance marks: paired tests against RATA-MAR (Holm-corrected within each compared model)
    tests = pd.read_csv(SRC / 'B_paired_tests.csv')
    if (SRC / 'B_paired_tests_baselines.csv').exists():
        tests = pd.concat([tests, pd.read_csv(SRC / 'B_paired_tests_baselines.csv')])

    def mark(name, c):
        r = tests[(tests.model_b == name.split(' ')[0]) & (tests.scenario == ('clean' if c == 'clean_top1' else c))]
        return '\\rlap{$^{\\ast}$}' if len(r) and r.p_holm.iloc[0] < 0.05 else ''
    for name, g, e in entries:
        cells = [ms(g[c], nd=1) + mark(name, c) if c in g.columns and g[c].notna().all() else '--' for c in cols_b]
        lines.append(f'{name} & ' + ' & '.join(cells) + f' & {e}\\\\')
    write('table_baselines_rows.tex', '\n'.join(lines))


if __name__ == '__main__':
    print('source of new analyses:', SRC)
    table_clean(); table_robust(); table_ablation(); table_stats(); table_extended(); table_detect(); table_variants()
