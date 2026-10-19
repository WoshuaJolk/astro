"""Aggregate ITF search results: local window logs (data/itf/run_small.log, fit logs) + worker-reported lines pasted into
data/itf/workers/*.txt (one file per worker, verbatim 'window NN MJD ...', 'window NN: ...', 'CANDIDATE', 'near-miss' lines)."""
import re, glob, os
D = '/home/ubuntu/iso/data/itf'
lines = []
for f in [f'{D}/run_small.log'] + glob.glob(f'{D}/fit_*.log') + glob.glob(f'{D}/workers/*.txt'):
    if os.path.exists(f): lines += [(os.path.basename(f), l.rstrip()) for l in open(f, errors='replace')]
summ = {}; fits = {}; cands = []; near = []
for src, l in lines:
    m = re.match(r'window (\d+) MJD (\S+): (\d+) tracklets, (\d+) solutions, (\d+) raw pairs, (\d+) unique.*total (\d+)s', l)
    if m: summ[int(m.group(1))] = (m.group(2), int(m.group(3)), int(m.group(5)), int(m.group(6)), int(m.group(7)), src)
    m = re.match(r'window (\d+): (\d+) chi<3 pairs, (\d+) per-cell clusters.*?(\d+) objects.*accepted (\d+)', l)
    if m: fits[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    if 'CANDIDATE' in l: cands.append((src, l.strip()))
    if 'near-miss' in l: near.append((src, l.strip()))
print(f'{"win":>3} {"MJD range":>12} {"tracklets":>9} {"raw pairs":>10} {"unique":>8} {"time s":>6} | {"chi<3":>8} {"clusters":>8} {"objects":>7} {"accepted":>8}  source')
for w in sorted(set(summ) | set(fits)):
    s = summ.get(w); f = fits.get(w)
    print(f'{w:3d} ' + (f'{s[0]:>12} {s[1]:9d} {s[2]:10d} {s[3]:8d} {s[4]:6d}' if s else ' ' * 49) + ' | ' + (f'{f[0]:8d} {f[1]:8d} {f[2]:7d} {f[3]:8d}' if f else ' ' * 34) + f'  {s[5] if s else ""}')
tot_tr = sum(s[1] for s in summ.values()); print(f'\n{len(summ)} windows with pair results ({tot_tr} tracklet-windows), {len(fits)} with fits, accepted candidates: {sum(f[3] for f in fits.values())}')
for src, l in cands: print('  ', src, l)
print(f'\nnear-misses (rms<10" but failed chi2/dof<100): {len(near)}')
for src, l in near[:40]: print('  ', src, l)
