import numpy as np
import matplotlib.pyplot as plt


def _save(path):
    if path is not None:
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_history(df, title='Training', save_path=None):
    plt.figure(figsize=(8,4))
    plt.plot(df['epoch'], df['val_top1'], label='val Top-1')
    plt.plot(df['epoch'], df['val_macro_f1'], label='val Macro-F1')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title(title)
    plt.grid(alpha=0.3); plt.legend(); _save(save_path); plt.show()


def plot_alignment(A, title='Skeleton → IMU alignment', save_path=None):
    A=np.asarray(A)
    plt.figure(figsize=(6,5)); plt.imshow(A,aspect='auto'); plt.colorbar()
    plt.xlabel('IMU token'); plt.ylabel('Skeleton query token'); plt.title(title)
    _save(save_path); plt.show()


def plot_shift_curve(df, title='Temporal-delay robustness', save_path=None):
    plt.figure(figsize=(7,4))
    for model_name,g in df.groupby('model'):
        g=g.sort_values('shift')
        plt.plot(g['shift'],g['top1'],marker='o',label=model_name)
    plt.xlabel('|IMU delay| in raw 64-step sequence'); plt.ylabel('Top-1 accuracy')
    plt.title(title); plt.grid(alpha=0.3); plt.legend(); _save(save_path); plt.show()


def plot_robustness(df, save_path=None):
    # df columns: scenario, model, top1
    piv=df.pivot(index='scenario',columns='model',values='top1')
    ax=piv.plot(kind='bar',figsize=(10,5))
    ax.set_ylabel('Top-1 accuracy'); ax.set_xlabel('Scenario'); ax.set_ylim(0,1)
    ax.grid(axis='y',alpha=0.25); plt.xticks(rotation=30,ha='right')
    _save(save_path); plt.show()
