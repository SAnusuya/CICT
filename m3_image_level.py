"""Image-level evaluation on membrane M3 (PVC-g-folic acid/Fe2O3), three independent micrographs.

Revision experiments (CICT 2026), complementing the spatial CV of porosity_cv.py:
  * Leave-one-image-out: train on two before-filtration micrographs, test on the third,
    which is never seen during training or checkpoint selection. Validation uses the last
    quarter (right-hand strip) of each training micrograph.
  * Same two target-mask protocols as porosity_cv.py (conservative / calibrated), built
    per micrograph from the M3 water-uptake porosity.
  * Fouling: models trained on all three before-filtration micrographs are applied to the
    three after-filtration micrographs.

Usage:  python m3_image_level.py --phi_before 75.0 [--phi_after 60.0] --data_dir ../data_fa_fe2o3
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from porosity_cv import (BATCH, EPOCHS, LR, PATCH, PATCHES_PER_EPOCH, CONSERVATIVE_OFFSET, VARIANTS,
                         bce, build_model, dice_iou, predict, target_mask, unsupervised_baselines)

BEFORE = ['FA_before_500x', 'FA_before_5kx', 'FA_before_4kx']
AFTER = ['FA_after_500x', 'FA_after_5kx', 'FA_after_4kx']


def split_train_val(img):
    """Training region = left three quarters, validation region = right quarter."""
    c = img.shape[1] * 3 // 4
    return img[:, :c], img[:, c:]


def sample_batch_multi(regions, rng):
    ims, ms = [], []
    for _ in range(BATCH):
        img, mask = regions[rng.integers(0, len(regions))]
        h, w = img.shape
        y, x = rng.integers(0, h - PATCH + 1), rng.integers(0, w - PATCH + 1)
        a, b = img[y:y + PATCH, x:x + PATCH], mask[y:y + PATCH, x:x + PATCH]
        k = rng.integers(0, 4)
        a, b = np.rot90(a, k), np.rot90(b, k)
        if rng.random() < 0.5:
            a, b = np.fliplr(a), np.fliplr(b)
        ims.append(a.copy())
        ms.append(b.copy())
    return (torch.from_numpy(np.stack(ims)[:, None].astype(np.float32) / 255.0),
            torch.from_numpy(np.stack(ms)[:, None].astype(np.float32)))


def train_multi(name, pairs, seed, device):
    """pairs: list of (img, mask) training micrographs."""
    train_regions, val_regions = [], []
    for img, m in pairs:
        (ti, vi), (tm, vm) = split_train_val(img), split_train_val(m)
        train_regions.append((ti, tm))
        val_regions.append((vi, vm))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    best_val, best_state, best_epoch = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        for _ in range(PATCHES_PER_EPOCH // BATCH):
            x, y = sample_batch_multi(train_regions, rng)
            loss = lossf(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = float(np.mean([bce(predict(model, vi, device), vm) for vi, vm in val_regions]))
        if vl < best_val:
            best_val, best_epoch = vl, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='../data_fa_fe2o3')
    ap.add_argument('--out_dir', default='../revision_results')
    ap.add_argument('--phi_before', type=float, required=True, help='M3 water-uptake porosity (%)')
    ap.add_argument('--phi_after', type=float, default=None, help='M3 porosity after filtration (%), if measured')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--models', nargs='+', default=list(VARIANTS))
    ap.add_argument('--tag', default='m3')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, 'masks_m3'), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_path = os.path.join(args.out_dir, f'results_{args.tag}.json')
    R = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def save():
        json.dump(R, open(res_path, 'w'), indent=1)

    imgs = {n: cv2.imread(os.path.join(args.data_dir, n + '.png'), 0) for n in BEFORE + AFTER}
    protos = {'conservative': args.phi_before - CONSERVATIVE_OFFSET, 'calibrated': args.phi_before}
    masks = {}
    R['meta'] = {'phi_before': args.phi_before, 'phi_after': args.phi_after, 'masks': {}}
    for n in BEFORE:
        for proto, tgt in protos.items():
            m, t = target_mask(imgs[n], tgt)
            masks[(n, proto)] = m
            R['meta']['masks'][f'{n}|{proto}'] = {'nominal': tgt, 'grey_threshold': t,
                                                  'measured': float(m.mean() * 100)}
    R['unsupervised'] = {n: unsupervised_baselines(imgs[n]) for n in BEFORE + AFTER}
    save()

    R.setdefault('loio', {})
    R.setdefault('fouling', {})
    for proto in protos:
        for name in args.models:
            for seed in args.seeds:
                # leave-one-image-out
                for i, test in enumerate(BEFORE):
                    key = f'{test}|{proto}|{name}|{seed}'
                    if key in R['loio']:
                        continue
                    t0 = time.time()
                    pairs = [(imgs[n], masks[(n, proto)]) for n in BEFORE if n != test]
                    model, be, bv = train_multi(name, pairs, seed * 100 + i, device)
                    p = predict(model, imgs[test], device)
                    pb = p >= 0.5
                    gt = masks[(test, proto)]
                    d, iou = dice_iou(pb, gt.astype(bool))
                    R['loio'][key] = {'porosity': float(pb.mean() * 100), 'dice': float(d), 'iou': float(iou),
                                      'test_loss': bce(p, gt), 'best_epoch': be, 'val_loss': bv,
                                      'mask_porosity': float(gt.mean() * 100), 'time_s': time.time() - t0}
                    if seed == 0:
                        cv2.imwrite(os.path.join(args.out_dir, 'masks_m3', f'{test}_{proto}_{name}.png'),
                                    (pb * 255).astype(np.uint8))
                    save()
                    print(f'LOIO {key}: phi={pb.mean()*100:.2f} dice={d:.4f} ({time.time()-t0:.0f}s)', flush=True)
                # fouling: train on all before images, predict after images
                key = f'{proto}|{name}|{seed}'
                if key in R['fouling']:
                    continue
                model, be, _ = train_multi(name, [(imgs[n], masks[(n, proto)]) for n in BEFORE],
                                           seed * 100 + 9, device)
                out = {}
                for n in BEFORE + AFTER:
                    pb = predict(model, imgs[n], device) >= 0.5
                    out[n] = float(pb.mean() * 100)
                    if seed == 0 and n in AFTER:
                        cv2.imwrite(os.path.join(args.out_dir, 'masks_m3', f'{n}_{proto}_{name}.png'),
                                    (pb * 255).astype(np.uint8))
                R['fouling'][key] = {'porosity': out, 'best_epoch': be}
                save()
                print(f'FOUL {key}: ' + ' '.join(f'{n}={v:.2f}' for n, v in out.items()), flush=True)


if __name__ == '__main__':
    main()
