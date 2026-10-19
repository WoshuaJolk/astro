import numpy as np, time, sys
from search import load_tracklets
import kernel4 as K4, search6b as S3
NAME = sys.argv[1]; NL = {'1I': 231195, '2I': 651864, '3I': 366346}[NAME]
tr = load_tracklets(f'/home/ubuntu/iso/data/test/inj_{NAME}.npz')
isiso = tr['first_line'] >= NL; print(f'{NAME}: {isiso.sum()} injected tracklets among {len(isiso)}')
g = K4.make_grid(2.5); cb = K4.cell_bases(g)
t0 = time.time(); P, over, nsol = S3.search(tr, g, cb, npass=int(sys.argv[2]) if len(sys.argv) > 2 else 4); nf = 0
print(f'total: {nsol} solutions, {len(P)} raw pairs, overflow {over}, narrow frac {nf*100:.1f}%, {time.time()-t0:.0f} s', flush=True)
P_all = P; P = S3.dedupe(P); print(f'pairs unique: {len(P)}'); np.savez(f'/home/ubuntu/iso/data/test/pairs_{NAME}.npz', P=P, P_all=P_all)
a = P[:, 0].astype(int); b = P[:, 1].astype(int)
ii = isiso[a] & isiso[b]; print(f'ISO-ISO pairs {ii.sum()}, ISO-other {(isiso[a] ^ isiso[b]).sum()}, other-other {(~isiso[a] & ~isiso[b]).sum()}')
niso = isiso.sum(); print(f'ISO tracklets appearing in an ISO-ISO pair: {len(np.unique(np.r_[a[ii], b[ii]]))}/{niso}')
for thr in (3.0, 20.0):
    Q = P[P[:, 3] < thr]; a2 = Q[:, 0].astype(int); b2 = Q[:, 1].astype(int); i2 = isiso[a2] & isiso[b2]
    print(f'chi<{thr}: pairs {len(Q)}, ISO-ISO {i2.sum()}, random {(~isiso[a2] & ~isiso[b2]).sum()}')
P3 = P_all[P_all[:, 3] < 3.0]
t0 = time.time(); cl = S3.triangle_clusters(P3, g['cell_rad']); print(f'triangle clusters (per cell, chi<3) >=3: {len(cl)} ({time.time()-t0:.0f}s)')
sizes = np.array([len(c[1]) for c in cl]); isoc = np.array([isiso[c[1]].sum() for c in cl])
print('  size distribution:', np.bincount(np.minimum(sizes, 10))[3:] if len(cl) else [], ' with >=3 ISO:', (isoc >= 3).sum(), ' max ISO members', isoc.max() if len(isoc) else 0, ' pure-random:', (isoc == 0).sum())
if (isoc == 0).sum(): print('  random cluster sizes:', np.bincount(sizes[isoc == 0])[3:])
print('merge/fit: run fitclusters.py', flush=True)
np.savez(f'/home/ubuntu/iso/data/test/pairs_{NAME}.npz', P=P, P_all=P_all)
