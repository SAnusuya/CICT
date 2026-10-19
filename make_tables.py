"""Print the result tables of the revised paper from results_seed*.json."""
import glob
import json
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else '../revision_results'
MODELS = ['Tiny', 'Standard', 'Wide', 'Shallow', 'Deep', 'ResNet18-UNet']
EXP = {'A1': 76.1, 'A2': 78.3}
CAP = {'Tiny': 0.5, 'Standard': 1.0, 'Wide': 2.0, 'Shallow': 0.93, 'Deep': 2.07}


def load():
    R = {}
    for f in sorted(glob.glob(os.path.join(RES, 'results_seed*.json'))):
        for k, v in json.load(open(f)).items():
            if k in R and isinstance(v, dict):
                R[k].update(v)
            else:
                R.setdefault(k, v)
    return R


def vals(D, prefix, field):
    return np.array([v[field] for k, v in D.items() if k.startswith(prefix)])


def ms(a, d=2):
    if len(a) == 0:
        return '–'
    return f'{a.mean():.{d}f} ± {a.std(ddof=1) if len(a) > 1 else 0:.{d}f}'


def main():
    R = load()
    print('== meta')
    for k, v in R['meta'].items():
        print(k, v)
    print('\n== unsupervised', R['unsupervised'])
    print('== rf (held-out strips)', R['rf'])

    for proto in ('calibrated', 'conservative'):
        print(f'\n== Held-out segmentation ({proto}): Dice | IoU | test BCE | best epoch  [n runs]')
        for m in MODELS:
            row = [m]
            for s in ('A1', 'A2'):
                p = f'{s}|{proto}|{m}|'
                d, i, l = vals(R['cv'], p, 'dice'), vals(R['cv'], p, 'iou'), vals(R['cv'], p, 'test_loss')
                be = np.array([f['best_epoch'] for k, v in R['cv'].items() if k.startswith(p) for f in v['folds']])
                row += [ms(d, 4), ms(i, 4), ms(l, 4), f'{be.mean():.0f}' if len(be) else '–', f'[{len(d)}]']
            print(' | '.join(row))

    print('\n== Held-out porosity (out-of-fold, whole image): mean ± SD over seeds; Δ vs water uptake; Δ vs mask')
    for proto in ('conservative', 'calibrated'):
        print(f'-- {proto}  (mask A1 {R["meta"]["A1_"+proto]["measured"]:.2f}, A2 {R["meta"]["A2_"+proto]["measured"]:.2f})')
        mus = {}
        for m in MODELS:
            row = [m]
            for s in ('A1', 'A2'):
                a = vals(R['cv'], f'{s}|{proto}|{m}|', 'porosity')
                if len(a) == 0:
                    row += ['–'] * 4
                    continue
                mus[(m, s)] = a.mean()
                row += [ms(a), f'{a.mean()-EXP[s]:+.2f}', f'{abs(a.mean()-EXP[s])/EXP[s]*100:.2f}',
                        f'{a.mean()-R["meta"][s+"_"+proto]["measured"]:+.2f}']
            if (m, 'A1') in mus and (m, 'A2') in mus:
                row.append(f'gap {mus[(m,"A2")]-mus[(m,"A1")]:+.2f}')
            print(' | '.join(row))
        for s in ('A1', 'A2'):
            ms_ = [m for m in CAP if (m, s) in mus]
            if len(ms_) == 5:
                r = np.corrcoef([CAP[m] for m in ms_], [mus[(m, s)] for m in ms_])[0, 1]
                spread = [mus[(m, s)] for m in MODELS if (m, s) in mus]
                print(f'   {s}: Pearson r(capacity, porosity) = {r:.2f}; range across models '
                      f'{min(spread):.2f}–{max(spread):.2f} ({max(spread)-min(spread):.2f} pp)')

    print('\n== Per-strip porosity vs mask (calibrated, seed-mean): strip | mask | ' + ' | '.join(MODELS))
    for s in ('A1', 'A2'):
        for k in range(4):
            row = [f'{s} S{k+1}', f'{R["meta"][s+"_calibrated"]["strip_fractions"][k]:.1f}']
            for m in MODELS:
                a = [f['porosity'] for kk, v in R['cv'].items() if kk.startswith(f'{s}|calibrated|{m}|')
                     for f in v['folds'] if f['fold'] == k]
                row.append(f'{np.mean(a):.1f}' if a else '–')
            print(' | '.join(row))

    if R.get('transfer'):
        print('\n== Transfer (train on source, test on whole target): porosity mean ± SD | Δ vs exp | Dice')
        for proto in ('conservative', 'calibrated'):
            for src, dst in (('A1', 'A2'), ('A2', 'A1')):
                print(f'-- {src}->{dst} {proto} (target mask {R["meta"][dst+"_"+proto]["measured"]:.2f}, exp {EXP[dst]})')
                for m in MODELS:
                    p = f'{src}->{dst}|{proto}|{m}|'
                    a, d = vals(R['transfer'], p, 'porosity'), vals(R['transfer'], p, 'dice')
                    if len(a):
                        print(f'   {m} | {ms(a)} | {a.mean()-EXP[dst]:+.2f} | {ms(d, 4)} [{len(a)}]')


if __name__ == '__main__':
    main()
