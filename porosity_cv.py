"""Spatially held-out evaluation of U-Net variants for SEM porosity estimation.

Revision experiments (CICT 2026):
  * 4-fold spatial cross-validation per micrograph: the image is split into four
    non-overlapping vertical strips; in each fold two strips are used for
    training, one for validation (checkpoint selection) and one for testing.
  * Two target-mask protocols applied identically to both membranes:
      conservative = experimental porosity - 5.1 pp (A1: 71.0 %, A2: 73.2 %)
      calibrated   = experimental porosity          (A1: 76.1 %, A2: 78.3 %)
  * Cross-membrane transfer: train on one micrograph, test on the other.
  * Baselines: Otsu, K-Means, GMM (unsupervised) and a pixel-wise random forest
    trained on the same training strips and tested on the held-out strip.
  * Five U-Net variants plus a U-Net with an ImageNet-pretrained ResNet-18 encoder.

Usage:  python porosity_cv.py --data_dir .. --out_dir ../revision_results
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

IMAGES = {'A1': ('a1.png', 76.1), 'A2': ('a2.png', 78.3)}
CONSERVATIVE_OFFSET = 5.1  # pp below experimental porosity (76.1 - 71.0 for A1)
N_STRIPS = 4
PATCH = 128
PATCHES_PER_EPOCH = 200
BATCH = 8
EPOCHS = 50
LR = 1e-3

VARIANTS = {
    'Tiny': [8, 16, 32, 64],
    'Standard': [16, 32, 64, 128],
    'Wide': [32, 64, 128, 256],
    'Shallow': [32, 64, 128],
    'Deep': [16, 32, 64, 128, 256],
    'ResNet18-UNet': None,  # ImageNet-pretrained encoder
}


# ----------------------------------------------------------------------------- models
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UNetVariant(nn.Module):
    """Same architecture as the original study (train_unet_variants.py); returns logits."""

    def __init__(self, features, cin=1):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(cin, f))
            cin = f
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i // 2 + 1)]
            if x.shape[2:] != s.shape[2:]:
                x = F.interpolate(x, size=s.shape[2:])
            x = self.ups[i + 1](torch.cat([s, x], 1))
        return self.final(x)


class ResNet18UNet(nn.Module):
    """U-Net decoder on an ImageNet-pretrained ResNet-18 encoder."""

    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)        # 64, /2
        self.pool = r.maxpool                                    # /4
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64,128,256,512
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def up(cin, cskip, cout):
            return nn.ModuleDict({'up': nn.ConvTranspose2d(cin, cout, 2, 2), 'conv': DoubleConv(cout + cskip, cout)})
        self.u4 = up(512, 256, 256)
        self.u3 = up(256, 128, 128)
        self.u2 = up(128, 64, 64)
        self.u1 = up(64, 64, 32)
        self.u0 = up(32, 1, 16)
        self.final = nn.Conv2d(16, 1, 1)

    @staticmethod
    def _up(block, x, skip):
        x = block['up'](x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:])
        return block['conv'](torch.cat([skip, x], 1))

    def forward(self, x):
        x0 = x
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        s1 = self.stem(x)
        e1 = self.l1(self.pool(s1))
        e2 = self.l2(e1)
        e3 = self.l3(e2)
        e4 = self.l4(e3)
        d = self._up(self.u4, e4, e3)
        d = self._up(self.u3, d, e2)
        d = self._up(self.u2, d, e1)
        d = self._up(self.u1, d, s1)
        d = self._up(self.u0, d, x0)
        return self.final(d)


def build_model(name):
    return ResNet18UNet() if VARIANTS[name] is None else UNetVariant(VARIANTS[name])


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- data
def target_mask(img, target_pct):
    """Intensity-ranking mask: darkest pixels up to the grey level whose cumulative
    fraction is closest to target_pct (same rule as the original study)."""
    vals = np.arange(256)
    cum = np.cumsum(np.bincount(img.ravel(), minlength=256)) / img.size * 100
    t = int(vals[np.argmin(np.abs(cum - target_pct))])
    return (img <= t).astype(np.uint8), t


def strip_bounds(w):
    edges = np.linspace(0, w, N_STRIPS + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(N_STRIPS)]


def fold_roles(k):
    """Fold k: test strip k, validation strip k+1, training strips the remaining two."""
    test, val = k, (k + 1) % N_STRIPS
    train = [s for s in range(N_STRIPS) if s not in (test, val)]
    return train, val, test


def valid_x_starts(bounds, strips, w):
    cols = np.zeros(w, bool)
    for s in strips:
        a, b = bounds[s]
        cols[a:b] = True
    return [x for x in range(w - PATCH + 1) if cols[x:x + PATCH].all()]


def sample_batch(img, mask, xs, rng):
    h = img.shape[0]
    ims, ms = [], []
    for _ in range(BATCH):
        x, y = rng.choice(xs), rng.integers(0, h - PATCH + 1)
        a, b = img[y:y + PATCH, x:x + PATCH], mask[y:y + PATCH, x:x + PATCH]
        k = rng.integers(0, 4)
        a, b = np.rot90(a, k), np.rot90(b, k)
        if rng.random() < 0.5:
            a, b = np.fliplr(a), np.fliplr(b)
        ims.append(a.copy())
        ms.append(b.copy())
    return (torch.from_numpy(np.stack(ims)[:, None].astype(np.float32) / 255.0),
            torch.from_numpy(np.stack(ms)[:, None].astype(np.float32)))


@torch.no_grad()
def predict(model, region, device):
    """Predict pore probability for a region; reflect-pad to a multiple of 32."""
    model.eval()
    h, w = region.shape
    ph, pw = (-h) % 32, (-w) % 32
    r = np.pad(region, ((0, ph), (0, pw)), mode='reflect').astype(np.float32) / 255.0
    out = torch.sigmoid(model(torch.from_numpy(r)[None, None].to(device)))
    return out[0, 0, :h, :w].cpu().numpy()


def bce(prob, mask):
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(-np.mean(mask * np.log(p) + (1 - mask) * np.log(1 - p)))


def dice_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    s = pred.sum() + gt.sum()
    union = np.logical_or(pred, gt).sum()
    return (2 * inter / s if s else 1.0), (inter / union if union else 1.0)


# ----------------------------------------------------------------------------- training
def train_one(name, img, mask, train_strips, val_strip, seed, device, log_curve=False):
    bounds = strip_bounds(img.shape[1])
    xs = valid_x_starts(bounds, train_strips, img.shape[1])
    va, vb = bounds[val_strip]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    best_val, best_state, best_epoch, curve = np.inf, None, 0, []
    for ep in range(EPOCHS):
        model.train()
        tl = 0.0
        for _ in range(PATCHES_PER_EPOCH // BATCH):
            x, y = sample_batch(img, mask, xs, rng)
            x, y = x.to(device), y.to(device)
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= PATCHES_PER_EPOCH // BATCH
        vl = bce(predict(model, img[:, va:vb], device), mask[:, va:vb])
        curve.append((tl, vl))
        if vl < best_val:
            best_val, best_epoch = vl, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, best_val, curve


# ----------------------------------------------------------------------------- baselines
def rf_features(img):
    f = img.astype(np.float32)
    return np.stack([f, cv2.GaussianBlur(f, (3, 3), 0), cv2.GaussianBlur(f, (7, 7), 0),
                     cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)], -1)


def unsupervised_baselines(img):
    out = {}
    t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out['Otsu'] = float(np.mean(img <= t) * 100)
    X = img.reshape(-1, 1).astype(np.float32)
    km = KMeans(2, n_init=10, random_state=42).fit(X)
    dark = np.argmin(km.cluster_centers_.ravel())
    out['K-Means (k=2)'] = float(np.mean(km.labels_ == dark) * 100)
    g = GaussianMixture(3, random_state=42).fit(X)
    lab = g.predict(X)
    order = np.argsort(g.means_.ravel())
    out['GMM (comp. 1)'] = float(np.mean(lab == order[0]) * 100)
    out['GMM (comp. 1+2)'] = float(np.mean((lab == order[0]) | (lab == order[1])) * 100)
    return out


def rf_cv(img, mask, seed):
    feats = rf_features(img)
    bounds = strip_bounds(img.shape[1])
    oof = np.zeros(img.shape, np.uint8)
    for k in range(N_STRIPS):
        train, _, test = fold_roles(k)
        cols = np.concatenate([np.arange(*bounds[s]) for s in train])
        X = feats[:, cols].reshape(-1, 5)
        y = mask[:, cols].ravel()
        idx = np.random.default_rng(seed).choice(len(y), min(40000, len(y)), replace=False)
        rf = RandomForestClassifier(80, max_depth=12, random_state=seed, n_jobs=-1).fit(X[idx], y[idx])
        a, b = bounds[test]
        p = rf.predict_proba(feats[:, a:b].reshape(-1, 5))[:, 1].reshape(img.shape[0], b - a)
        oof[:, a:b] = p >= 0.5
    return oof


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='..')
    ap.add_argument('--out_dir', default='../revision_results')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--models', nargs='+', default=list(VARIANTS))
    ap.add_argument('--tag', default='all', help='results file suffix (lets seeds run in parallel)')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, 'masks'), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_path = os.path.join(args.out_dir, f'results_{args.tag}.json')
    R = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def save():
        json.dump(R, open(res_path, 'w'), indent=1)

    imgs, masks = {}, {}
    R.setdefault('meta', {})
    for s, (fn, exp) in IMAGES.items():
        img = cv2.imread(os.path.join(args.data_dir, fn), 0)
        imgs[s] = img
        for proto, tgt in (('conservative', exp - CONSERVATIVE_OFFSET), ('calibrated', exp)):
            m, t = target_mask(img, tgt)
            masks[(s, proto)] = m
            R['meta'][f'{s}_{proto}'] = {'nominal': tgt, 'grey_threshold': t,
                                         'measured': float(m.mean() * 100),
                                         'strip_fractions': [float(m[:, a:b].mean() * 100)
                                                             for a, b in strip_bounds(img.shape[1])]}
        R['meta'][s] = {'shape': list(img.shape), 'experimental': exp, 'strips': strip_bounds(img.shape[1])}
    R['meta']['params'] = {n: n_params(build_model(n)) for n in VARIANTS}
    R.setdefault('unsupervised', {s: unsupervised_baselines(imgs[s]) for s in IMAGES})
    save()

    # random forest, same spatial CV
    R.setdefault('rf', {})
    for (s, proto), m in masks.items():
        key = f'{s}|{proto}'
        if key in R['rf']:
            continue
        oof = rf_cv(imgs[s], m, 0)
        d, i = dice_iou(oof.astype(bool), m.astype(bool))
        R['rf'][key] = {'porosity': float(oof.mean() * 100), 'dice': float(d), 'iou': float(i)}
        save()

    # U-Net spatial CV
    R.setdefault('cv', {})
    R.setdefault('curves', {})
    for proto in ('conservative', 'calibrated'):
        for s in IMAGES:
            img, m = imgs[s], masks[(s, proto)]
            bounds = strip_bounds(img.shape[1])
            for name in args.models:
                for seed in args.seeds:
                    key = f'{s}|{proto}|{name}|{seed}'
                    if key in R['cv']:
                        continue
                    t0 = time.time()
                    oof_prob = np.zeros(img.shape, np.float32)
                    folds = []
                    for k in range(N_STRIPS):
                        train, val, test = fold_roles(k)
                        model, be, bv, curve = train_one(name, img, m, train, val, seed * 100 + k, device)
                        if seed == 0:
                            R['curves'][f'{s}|{proto}|{name}|fold{k}'] = curve
                        a, b = bounds[test]
                        p = predict(model, img[:, a:b], device)
                        oof_prob[:, a:b] = p
                        pb = p >= 0.5
                        d, i = dice_iou(pb, m[:, a:b].astype(bool))
                        folds.append({'fold': k, 'best_epoch': be, 'val_loss': bv,
                                      'test_loss': bce(p, m[:, a:b]), 'dice': float(d), 'iou': float(i),
                                      'porosity': float(pb.mean() * 100),
                                      'mask_porosity': float(m[:, a:b].mean() * 100)})
                    oof = oof_prob >= 0.5
                    d, i = dice_iou(oof, m.astype(bool))
                    R['cv'][key] = {'folds': folds, 'porosity': float(oof.mean() * 100),
                                    'dice': float(d), 'iou': float(i), 'test_loss': bce(oof_prob, m),
                                    'time_s': time.time() - t0}
                    if seed == 0:
                        cv2.imwrite(os.path.join(args.out_dir, 'masks', f'{s}_{proto}_{name}.png'),
                                    (oof * 255).astype(np.uint8))
                    save()
                    print(f'{key}: phi={oof.mean()*100:.2f}  dice={d:.4f}  ({time.time()-t0:.0f}s)', flush=True)

    # cross-membrane transfer: train on all of source (strips 0,1,2 train; 3 val), test on full target
    R.setdefault('transfer', {})
    for proto in ('conservative', 'calibrated'):
        for src, dst in (('A1', 'A2'), ('A2', 'A1')):
            for name in args.models:
                for seed in args.seeds:
                    key = f'{src}->{dst}|{proto}|{name}|{seed}'
                    if key in R['transfer']:
                        continue
                    model, be, bv, _ = train_one(name, imgs[src], masks[(src, proto)], [0, 1, 2], 3,
                                                 seed * 100 + 7, device)
                    p = predict(model, imgs[dst], device)
                    pb = p >= 0.5
                    gt = masks[(dst, proto)].astype(bool)
                    d, i = dice_iou(pb, gt)
                    R['transfer'][key] = {'porosity': float(pb.mean() * 100), 'dice': float(d),
                                          'iou': float(i), 'best_epoch': be}
                    save()
                    print(f'{key}: phi={pb.mean()*100:.2f}  dice={d:.4f}', flush=True)


if __name__ == '__main__':
    main()
