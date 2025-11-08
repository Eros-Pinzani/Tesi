import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

ICPRootDir = os.path.join('img', 'icp')

# Utility salvataggio

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _savefig(path: str, dpi: int = 140):
    _ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


# 1) Schema concettuale corrispondenze (subset)

def save_concept_correspondences(res: Dict, title: str, out_path: str, max_lines: int = 120):
    tgt = np.asarray(res['tgt_local'])
    src = np.asarray(res['src_local'])
    if tgt.size == 0 or src.size == 0:
        return
    # Costruisci NN su subset semplice (euclideo O(N*M) sul subset)
    N = min(len(src), len(tgt), max_lines)
    src_sub = src[:N]
    # Abbina naive: per ogni src, trova NN su tgt
    idxs = []
    for p in src_sub:
        d2 = np.sum((tgt - p) ** 2, axis=1)
        idxs.append(int(np.argmin(d2)))
    plt.figure(figsize=(5, 4))
    plt.scatter(tgt[:, 0], tgt[:, 1], s=10, c='tab:blue', label='Target (k-1)')
    plt.scatter(src[:, 0], src[:, 1], s=10, c='tab:red', label='Source (k)')
    for i, j in enumerate(idxs):
        xs = [src_sub[i, 0], tgt[j, 0]]
        ys = [src_sub[i, 1], tgt[j, 1]]
        plt.plot(xs, ys, c='gray', lw=0.6, alpha=0.7)
    plt.axis('equal'); plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend(loc='upper right', fontsize=8)
    _savefig(out_path)


# 2) Effetto dell'inizializzazione: overlay finali

def save_alignment_overlays(res: Dict, title: str, out_path: str):
    tgt = np.asarray(res['tgt_local'])
    src_none = np.asarray(res['none']['src_transformed'])
    src_odo = np.asarray(res['odo']['src_transformed'])
    src_raw_none = np.asarray(res['raw_none']['src_transformed'])
    src_raw_odo = np.asarray(res['raw_odo']['src_transformed'])
    plt.figure(figsize=(6, 5))
    plt.scatter(tgt[:, 0], tgt[:, 1], s=10, c='k', label='Target (k-1)')
    plt.scatter(src_none[:, 0], src_none[:, 1], s=8, c='tab:red', alpha=0.7, label='ICP None')
    plt.scatter(src_odo[:, 0], src_odo[:, 1], s=8, c='tab:green', alpha=0.7, label='ICP Odo')
    plt.scatter(src_raw_none[:, 0], src_raw_none[:, 1], s=8, c='tab:orange', alpha=0.5, label='RAW None')
    plt.scatter(src_raw_odo[:, 0], src_raw_odo[:, 1], s=8, c='tab:purple', alpha=0.5, label='RAW Odo')
    plt.axis('equal'); plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend(loc='upper right', fontsize=8)
    _savefig(out_path)


# 3) Curve di convergenza (RMSE per iterazione)

def save_convergence_curves(res: Dict, title: str, out_path: str):
    e_none = np.asarray(res['none']['errs'])
    e_odo = np.asarray(res['odo']['errs'])
    e_rawn = np.asarray(res['raw_none']['errs'])
    e_rawo = np.asarray(res['raw_odo']['errs'])
    plt.figure(figsize=(6, 4))
    if e_none.size: plt.plot(e_none, label='ICP None')
    if e_odo.size: plt.plot(e_odo, label='ICP Odo')
    if e_rawn.size: plt.plot(e_rawn, '--', label='RAW None')
    if e_rawo.size: plt.plot(e_rawo, '--', label='RAW Odo')
    plt.xlabel('Iterazione'); plt.ylabel('RMSE')
    plt.title(title)
    plt.grid(alpha=0.3); plt.legend()
    _savefig(out_path)


# 9) Frecce GT vs stima (Δx, Δy) e α

def save_motion_arrows(res: Dict, title: str, out_path: str):
    # Frame locale di k-1
    def ang_deg(R):
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    gt_t = np.asarray(res['gt_t']); gt_R = np.asarray(res['gt_R'])
    ests = [
        ('None', res['none']['t'], res['none']['R'], 'tab:red'),
        ('Odo', res['odo']['t'], res['odo']['R'], 'tab:green'),
        ('RAW None', res['raw_none']['t'], res['raw_none']['R'], 'tab:orange'),
        ('RAW Odo', res['raw_odo']['t'], res['raw_odo']['R'], 'tab:purple'),
    ]
    plt.figure(figsize=(5, 4))
    # Ground truth
    plt.quiver(0, 0, gt_t[0], gt_t[1], angles='xy', scale_units='xy', scale=1, color='k', width=0.005, label=f'GT (α={ang_deg(gt_R):+.2f}°)')
    for name, t, R, col in ests:
        t = np.asarray(t)
        plt.quiver(0, 0, t[0], t[1], angles='xy', scale_units='xy', scale=1, color=col, width=0.004, label=f'{name} (α={ang_deg(np.asarray(R)):+.2f}°)')
    plt.axis('equal'); plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend(loc='upper right', fontsize=8)
    _savefig(out_path)


# 14) Confronto RAW vs Filtrato (overlay target + raw vs filtrato, per una coppia)

def save_raw_vs_filtered(res: Dict, title: str, out_path: str):
    tgt = np.asarray(res['tgt_local'])
    raw = np.asarray(res['raw_none']['src_transformed'])
    filt = np.asarray(res['none']['src_transformed'])
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(tgt[:, 0], tgt[:, 1], s=10, c='k', label='Target (k-1)')
    plt.scatter(raw[:, 0], raw[:, 1], s=8, c='tab:orange', alpha=0.6, label='RAW None')
    plt.scatter(filt[:, 0], filt[:, 1], s=8, c='tab:red', alpha=0.8, label='ICP None (filtrato)')
    plt.axis('equal'); plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend(loc='upper right', fontsize=8)
    _savefig(out_path)
