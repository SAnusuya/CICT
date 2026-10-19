"""Figures for the revised manuscript, drawn at IEEE single-column width (3.5 in), 600 dpi."""
import glob
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else '..'
RES = sys.argv[2] if len(sys.argv) > 2 else '../revision_results'
OUT = os.path.join(RES, 'figures')
os.makedirs(OUT, exist_ok=True)
W = 3.5
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8,
                     'axes.labelsize': 7.5, 'legend.fontsize': 6.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
MODELS = ['Tiny', 'Standard', 'Wide', 'Shallow', 'Deep', 'ResNet18-UNet']
SHORT = {'ResNet18-UNet': 'ResNet18'}
COL = {'Tiny': '#2a78d6', 'Standard': '#e8693a', 'Wide': '#1aa77a', 'Shallow': '#e9a100',
       'Deep': '#d6558f', 'ResNet18-UNet': '#6b5bc9'}
EXP = {'A1': 76.1, 'A2': 78.3}
STRIP_ROLE_TXT = ['S1', 'S2', 'S3', 'S4']


def fig_framework():
    steps = [('Membranes', 'M1: drug-PVC\nM2: drug-PVC/Fe$_2$O$_3$'),
             ('Water-uptake\nporosity', 'M1 = 76.1 %,  M2 = 78.3 %\n(physical reference)'),
             ('SEM imaging', 'A1 601×406 px, A2 601×407 px\npre-filtration'),
             ('Target masks', 'intensity ranking; conservative\n(φ$_{exp}$−5.1 pp) and calibrated (φ$_{exp}$)'),
             ('Spatial\n4-fold CV', '4 strips/image: 2 train, 1 val,\n1 test; 3 seeds; transfer A1↔A2'),
             ('Models', '5 U-Nets, pretrained ResNet18-UNet;\nOtsu, K-Means, GMM, RF'),
             ('Evaluation\n(held-out)', 'Dice, IoU vs mask;  Δ, E$_{rel}$ vs\nwater-uptake porosity')]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps))
    ax.axis('off')
    cols = ['#2a6fb0', '#c05621', '#2e8659', '#6b46c1', '#b7791f', '#b83280', '#4a5568']
    for i, ((t, d), c) in enumerate(zip(steps, cols)):
        y = len(steps) - i - 0.88
        ax.add_patch(FancyBboxPatch((0.01, y), 0.30, 0.70, boxstyle='round,pad=0,rounding_size=0.04',
                                    fc=c, ec=c))
        ax.text(0.16, y + 0.35, t, ha='center', va='center', color='white', fontsize=6.3, weight='bold',
                linespacing=1.0)
        ax.add_patch(FancyBboxPatch((0.34, y), 0.65, 0.70, boxstyle='round,pad=0,rounding_size=0.04',
                                    fc=matplotlib.colors.to_rgba(c, 0.12), ec=c, lw=0.8))
        ax.text(0.665, y + 0.35, d, ha='center', va='center', fontsize=6.1, linespacing=1.15)
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.16, y - 0.29), xytext=(0.16, y - 0.01),
                        arrowprops=dict(arrowstyle='-|>', color='#333', lw=0.9))
    fig.savefig(os.path.join(OUT, 'fig1_framework.png'))
    plt.close(fig)


def fig_sem():
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.42))
    for ax, (s, fn, lab) in zip(axes, [('A1', 'a1.png', '(a) A1 — M1, drug-PVC'),
                                        ('A2', 'a2.png', '(b) A2 — M2, drug-PVC/Fe$_2$O$_3$')]):
        img = cv2.imread(os.path.join(DATA, fn), 0)
        h, w = img.shape
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        edges = np.linspace(0, w, 5).astype(int)
        for k in range(4):
            if k:
                ax.axvline(edges[k] - 0.5, color='#ffd400', lw=1.0, ls='--')
            ax.text((edges[k] + edges[k + 1]) / 2, 14, STRIP_ROLE_TXT[k], color='black', ha='center',
                    va='top', fontsize=6, weight='bold',
                    bbox=dict(fc='#ffd400', ec='none', pad=1.2, alpha=0.9))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
        ax.set_xlabel(lab, fontsize=7, labelpad=2)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(os.path.join(OUT, 'fig2_sem.png'))
    plt.close(fig)


def load_results():
    R = {}
    for f in sorted(glob.glob(os.path.join(RES, 'results_seed*.json'))):
        d = json.load(open(f))
        for k, v in d.items():
            if isinstance(v, dict) and k in R and isinstance(R[k], dict):
                R[k].update(v)
            else:
                R.setdefault(k, v)
    return R


def fig_curves(R):
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.75), sharey=False)
    for ax, s in zip(axes, ['A1', 'A2']):
        for m in MODELS:
            key = f'{s}|calibrated|{m}|fold0'
            if key not in R['curves']:
                continue
            c = np.array(R['curves'][key])
            ep = np.arange(1, len(c) + 1)
            ax.plot(ep, c[:, 0], color=COL[m], lw=0.8, label=SHORT.get(m, m))
            ax.plot(ep, c[:, 1], color=COL[m], lw=0.8, ls=':')
        ax.set_title(f'{s} (fold 1, calibrated)', fontsize=7.5)
        ax.set_xlabel('Epoch')
        ax.set_ylim(0, 0.5)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel('BCE loss')
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color='k', lw=0.8), plt.Line2D([], [], color='k', lw=0.8, ls=':')]
    l += ['train', 'validation']
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2),
               fontsize=6.3, handlelength=1.5, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_curves.png'))
    plt.close(fig)


def stat(R, s, proto, m, field='porosity'):
    v = [R['cv'][k][field] for k in R['cv'] if k.startswith(f'{s}|{proto}|{m}|')]
    return np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v)


def fig_masks(R):
    fig, axes = plt.subplots(4, 4, figsize=(W, 2.95))
    for r0, s in ((0, 'A1'), (2, 'A2')):
        img = cv2.imread(os.path.join(DATA, s.lower() + '.png'), 0)
        meta = R['meta'][f'{s}_calibrated']
        gt = img <= meta['grey_threshold']
        tiles = [(img, 'SEM ' + s, 'gray'), (gt, f'Target {meta["measured"]:.1f}%', 'gray_r')]
        for m in MODELS:
            p = cv2.imread(os.path.join(RES, 'masks', f'{s}_calibrated_{m}.png'), 0) > 127
            tiles.append((p, f'{SHORT.get(m, m)} {p.mean()*100:.1f}%', 'gray_r'))
        for i, (t, title, cm) in enumerate(tiles):
            ax = axes[r0 + i // 4, i % 4]
            ax.imshow(t, cmap=cm, interpolation='nearest')
            ax.set_title(title, fontsize=6, pad=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.4)
    fig.tight_layout(pad=0.2, h_pad=0.35, w_pad=0.2)
    fig.savefig(os.path.join(OUT, 'fig4_masks.png'))
    plt.close(fig)


def fig_porosity(R):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15), sharey=True)
    y = np.arange(len(MODELS))
    for ax, s in zip(axes, ['A1', 'A2']):
        for j, (proto, mk, off) in enumerate((('conservative', 'o', -0.17), ('calibrated', 's', 0.17))):
            mu = [stat(R, s, proto, m)[0] for m in MODELS]
            sd = [stat(R, s, proto, m)[1] for m in MODELS]
            ax.errorbar(mu, y + off, xerr=sd, fmt=mk, ms=3.2, lw=0.8, capsize=1.8,
                        color=['#7f8c8d', '#1f4e79'][j], label=f'{proto} mask')
            ax.axvline(R['meta'][f'{s}_{proto}']['measured'], color=['#7f8c8d', '#1f4e79'][j],
                       lw=0.7, ls=':')
        ax.axvline(EXP[s], color='#c0392b', lw=1.0, ls='--', label='water uptake')
        ax.set_title(s, fontsize=8)
        ax.set_xlabel('Held-out porosity (%)')
        ax.grid(axis='x', alpha=0.3, lw=0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([SHORT.get(m, m) for m in MODELS])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.1),
               fontsize=6.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_porosity.png'))
    plt.close(fig)


if __name__ == '__main__':
    fig_framework()
    fig_sem()
    if glob.glob(os.path.join(RES, 'results_seed*.json')):
        R = load_results()
        if R.get('curves'):
            fig_curves(R)
        if R.get('cv') and len(R['cv']) >= 2 * 2 * len(MODELS):
            fig_masks(R)
            fig_porosity(R)
