"""Figures for Revision 1 (300 dpi PNG, written into manuscript_R1/).

  Fig2.png            development-split learning curves
  Fig3.png            robustness bars with SD error bars (published per-seed numbers, unchanged)
  Fig4.png            (a) accuracy vs IMU delay sweep  (b) attended token offset vs delay
                      (c, d) mean skeleton->IMU attention maps at delay 0 and +16

Usage: python -I r1_figures.py [--device cpu|cuda]
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get('RATA_PROJECT', Path(__file__).resolve().parents[2]))  # repository root
RUNS = PROJECT / 'runs/RATA_MAR_SI_v21'
MS = PROJECT / 'REVISION_R1/manuscript_R1'
MS.mkdir(parents=True, exist_ok=True)
OUT = PROJECT / 'REVISION_R1/experiments/outputs'

# fixed categorical order (validated palette); colour follows the model, never its rank
STYLE = {
    'RATA-MAR':          dict(color='#2a78d6', marker='o', ls='-'),
    'LateFusion-Robust': dict(color='#eb6834', marker='s', ls='--'),
    'LateFusion-Clean':  dict(color='#1baf7a', marker='^', ls=':'),
    'NoAlignment':       dict(color='#eda100', marker='D', ls='-.'),
    'NoReliability':     dict(color='#e87ba4', marker='v', ls=(0, (5, 1))),
}
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3df'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.dpi': 300, 'lines.linewidth': 2.0, 'lines.markersize': 5,
})


def panel(ax, letter):
    ax.text(-0.12, 1.04, f'({letter})', transform=ax.transAxes, fontsize=10, fontweight='bold', color=INK)


def fig_robustness():
    d = pd.read_csv(RUNS / 'tables/FINAL_XSUB_per_seed.csv')
    d['model'] = d.model.replace({'RATA-MAR-SI-v2.1': 'RATA-MAR'})
    scen = [('clean_top1', 'Clean'), ('missing_skeleton', 'Missing\nskeleton'), ('missing_imu', 'Missing\nIMU'),
            ('skeleton_noise', 'Skeleton\nnoise'), ('imu_noise', 'IMU\nnoise'), ('joint_drop_30', 'Joint\ndropout'),
            ('delay_pm16', 'Delay\n±16')]
    models = ['LateFusion-Clean', 'LateFusion-Robust', 'RATA-MAR']
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    w, gap = 0.26, 0.02
    x = np.arange(len(scen))
    for k, m in enumerate(models):
        g = d[d.model == m]
        mean = [g[c].mean() * 100 for c, _ in scen]
        sd = [g[c].std(ddof=1) * 100 for c, _ in scen]
        pos = x + (k - 1) * (w + gap)
        ax.bar(pos, mean, w, color=STYLE[m]['color'], label=m, edgecolor='white', linewidth=0.8,
               hatch={'LateFusion-Clean': '', 'LateFusion-Robust': '////', 'RATA-MAR': ''}[m])
        ax.errorbar(pos, mean, yerr=sd, fmt='none', ecolor=INK, elinewidth=0.8, capsize=2)
    ax.set_xticks(x, [s for _, s in scen])
    ax.set_ylabel('Top-1 accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.13), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(MS / 'Fig3.png', bbox_inches='tight')
    plt.close(fig)


def fig_dev_curves():
    runs = {'RATA-MAR': 'P1_RATA_MAR_SI_v21', 'LateFusion-Robust': 'B4_LateFusion_Robust',
            'LateFusion-Clean': 'B3_LateFusion_Clean', 'NoAlignment': 'A1_NoAlignment',
            'NoReliability': 'A2_NoReliability'}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for m, r in runs.items():
        h = pd.read_csv(RUNS / r / 'history.csv')
        st = STYLE[m]
        axes[0].plot(h.epoch, h.train_loss, color=st['color'], ls=st['ls'], lw=1.6, label=m)
        axes[1].plot(h.epoch, h.val_top1 * 100, color=st['color'], ls=st['ls'], lw=1.6, label=m)
        b = h.val_top1.idxmax()
        axes[1].plot(h.epoch[b], h.val_top1[b] * 100, marker=st['marker'], color=st['color'],
                     markersize=6, markeredgecolor='white', markeredgewidth=0.8, ls='none')
    axes[0].set(xlabel='Epoch', ylabel='Training objective', yscale='log')
    axes[1].set(xlabel='Epoch', ylabel='Validation top-1 (%)', ylim=(40, 100))
    panel(axes[0], 'a'); panel(axes[1], 'b')
    h0, l0 = axes[1].get_legend_handles_labels()
    fig.legend(h0, l0, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=8)
    fig.tight_layout()
    fig.savefig(MS / 'Fig2.png', bbox_inches='tight')
    plt.close(fig)


def fig_delay(dev):
    src = OUT / dev
    d = pd.read_csv(src / 'D_delay_sweep.csv')
    a = pd.read_csv(src / 'D_attention_offset.csv')
    maps = np.load(src / 'D_attention_maps_seed0.npz')
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    imu = d[d.stream == 'IMU']
    for m in ['LateFusion-Clean', 'LateFusion-Robust', 'NoAlignment', 'RATA-MAR']:
        g = imu[imu.model == m].groupby('shift').top1
        mu, sd = g.mean() * 100, g.std(ddof=1) * 100
        st = STYLE[m]
        ax.fill_between(mu.index, mu - sd, mu + sd, color=st['color'], alpha=0.12, lw=0)
        ax.plot(mu.index, mu.values, color=st['color'], ls=st['ls'], marker=st['marker'], markersize=4, label=m)
    for v in (-16, 16):
        ax.axvline(v, color=MUTED, lw=0.6, ls=':')
    ax.set(xlabel='IMU shift (time steps; positive = delayed)', ylabel='Top-1 accuracy (%)')
    ax.legend(fontsize=7.5, loc='lower center')
    panel(ax, 'a')

    ax = fig.add_subplot(gs[0, 1])
    g = a.groupby('shift')
    mu, sd = g.expected_offset_tokens.mean(), g.expected_offset_tokens.std(ddof=1)
    ax.plot(mu.index, mu.index / 8.0, color=MUTED, lw=1.0, ls='--', label='Ideal (shift / 8 steps per token)')
    ax.fill_between(mu.index, mu - sd, mu + sd, color=STYLE['RATA-MAR']['color'], alpha=0.15, lw=0)
    ax.plot(mu.index, mu.values, color=STYLE['RATA-MAR']['color'], marker='o', markersize=4,
            label='RATA-MAR (mean ± SD, 3 seeds)')
    ax.set(xlabel='IMU shift (time steps)', ylabel='Attended offset (tokens)')
    ax.legend(fontsize=7.5, loc='upper left')
    panel(ax, 'b')

    vmax = max(maps['shift_0'].max(), maps['shift_16'].max())
    for k, (key, title) in enumerate((('shift_0', 'IMU shift 0'), ('shift_16', 'IMU shift +16'))):
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(maps[key], cmap='Blues', vmin=0, vmax=vmax)
        ax.set(xlabel='IMU key token', ylabel='Skeleton query token', title=title)
        ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.grid(False)
        ax.title.set_fontsize(9)
        panel(ax, 'cd'[k])
    fig.colorbar(im, ax=fig.axes[-2:], fraction=0.025, pad=0.02, label='Mean attention weight')
    fig.savefig(MS / 'Fig4.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    if not args.only or 'rob' in args.only:
        fig_robustness()
    if not args.only or 'dev' in args.only:
        fig_dev_curves()
    if (not args.only or 'delay' in args.only) and (OUT / args.device / 'D_delay_sweep.csv').exists():
        fig_delay(args.device)
    print('figures written to', MS)
