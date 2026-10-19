"""Stage-3 v2: seed-and-grow instead of shared-member chaining.
Per connected group of per-cell clusters: fit each of the largest clusters ALONE (seed orbit), then grow each accepted
orbit by re-testing every leftover tracklet of the group against it; finally re-test every tracklet in the window
against each accepted orbit (recovers fragments of dense true objects without chaining unrelated clusters).
Thresholds identical to fitclusters.py (rms<10", chi2/dof<100, >=3 kept)."""
import numpy as np, sys, time
from search import load_tracklets
import kernel4 as K4, search6b as S4, fit2
NAME = sys.argv[1]; NL = {'1I': 231195, '2I': 651864, '3I': 366346}[NAME]
tr = load_tracklets(f'/home/ubuntu/iso/data/test/inj_{NAME}.npz'); isiso = tr['first_line'] >= NL
g = K4.make_grid(2.5); cb = K4.cell_bases(g)
d = np.load(f'/home/ubuntu/iso/data/test/pairs_{NAME}.npz'); P_all = d['P_all']
cl = S4.triangle_clusters(P_all[P_all[:, 3] < 3.0], g['cell_rad'])
t0 = time.time(); owner = {}
for i, c in enumerate(cl):
    for m in c[1]: owner.setdefault(int(m), []).append(i)
from collections import Counter
cnt = Counter()
for m, lst in owner.items():
    for a in range(len(lst)):
        for b in range(a + 1, len(lst)): cnt[(lst[a], lst[b])] += 1
E = np.array([e for e, k in cnt.items() if k >= 2]) if cnt else np.zeros((0, 2), int)
ccomp = S4.components(E if len(E) else np.zeros((0, 2)), len(cl)); groups = {}
for i, c in enumerate(cl): groups.setdefault(ccomp[i], []).append(i)
print(f'{len(cl)} per-cell clusters -> {len(groups)} groups (merge {time.time()-t0:.1f}s)', flush=True)

ALL = np.arange(len(tr['t'])); tmp = np.empty(6)
def grow(kept, x, te, pool, rms, c2=0.0):
    """time-expanding growth: consider pool tracklets within one current arc-span of the arc ends, add those with
    position residual < 30", refit with outlier rejection at max(3 rms, 2"); accept the round only if >=90% of previous
    members survive and rms<10", chi2/dof<100. Repeat until no additions."""
    for _ in range(30):
        pool = np.setdiff1d(pool, kept)
        if len(pool) == 0: break
        tmin, tmax = tr['t'][kept].min(), tr['t'][kept].max(); span = max(tmax - tmin, 10.0)
        sel = pool[(tr['t'][pool] > tmin - span) & (tr['t'][pool] < tmax + span)]
        if len(sel) == 0: break
        res = np.empty((len(sel), 4))
        if fit2.residuals(x, te, tr['t'][sel], tr['u'][sel], tr['udot'][sel], tr['obs_pos'][sel], tr['obs_vel'][sel], res, tmp) == 0: break
        pos = np.hypot(res[:, 0], res[:, 1]) * 206264.806
        add = sel[pos < 30.0]
        if len(add) == 0: break
        k2, rms2, c22, x2, te2 = fit2.robust_fit(tr, np.union1d(kept, add), [(x, te)], thr_arcsec=max(3.0 * rms, 2.0))
        if not (np.isfinite(rms2) and rms2 < 10.0 and c22 < 100.0 and len(np.intersect1d(kept, k2)) >= 0.9 * len(kept) and len(k2) > len(kept)): break
        kept, x, te, rms, c2 = k2, x2, te2, rms2, c22
    return kept, x, te, rms, c2

t0 = time.time(); acc = []; rows = []; nfit = 0
for gid, cis in groups.items():
    gmem = np.unique(np.concatenate([cl[i][1] for i in cis])).astype(int)
    top = sorted(cis, key=lambda i: -len(cl[i][1]))[:12]
    done = np.zeros(len(tr['t']), bool)
    for ci in top:
        cmem = np.asarray(cl[ci][1], int)
        if done[cmem].sum() >= 0.5 * len(cmem): continue      # already explained by an accepted orbit
        sub = {f: tr[f][cmem] for f in tr}
        cell, tid, val = S4.run_kernel(sub, g, cb, 0, g['nb'])
        m = np.where(cell == cl[ci][0])[0]
        if len(m) == 0: continue
        # seeds: best-sigma solution plus two more distinct members (multiple roots / fragments)
        order = m[np.argsort(val[m, 17])]; seeds = []; used = set()
        for j in order:
            if tid[j] in used: continue
            used.add(tid[j]); seeds.append(fit2.initial_state(tr, cmem[tid[j]], val[j, 7], val[j, 8]))
            if len(seeds) == 3: break
        kept, rms, c2, x, te = fit2.robust_fit(tr, cmem, seeds); nfit += 1
        rows.append((len(cmem), int(isiso[cmem].sum()), len(kept), int(isiso[kept].sum()), rms, c2))
        if not (np.isfinite(rms) and rms < 10.0 and c2 < 100.0 and len(kept) >= 3): continue
        kept, x, te, rms, c2 = grow(kept, x, te, gmem, rms, c2)
        done[kept] = True
        acc.append([kept, x, te, rms, c2])
print(f'seed fits: {nfit} in {time.time()-t0:.1f}s; accepted seeds {len(acc)}', flush=True)
rnd = np.array([(r[4], r[5], r[2]) for r in rows if r[1] == 0 and np.isfinite(r[4])])
if len(rnd): print(f'random seed clusters: {len(rnd)}; n with rms<10"', (rnd[:, 0] < 10).sum(), ' <10" & kept>=4', ((rnd[:, 0] < 10) & (rnd[:, 2] >= 4)).sum())
# window-wide grow + dedupe (objects sharing >=50% of the smaller one's members -> keep the larger/better)
t0 = time.time()
for a in acc:
    a[0], a[1], a[2], a[3], a[4] = grow(a[0], a[1], a[2], ALL, a[3], a[4])
acc.sort(key=lambda a: (-len(a[0]), a[3])); final = []
for a in acc:
    if any(len(np.intersect1d(a[0], f[0])) >= 0.5 * min(len(a[0]), len(f[0])) for f in final): continue
    final.append(a)
print(f'window grow+dedupe {time.time()-t0:.1f}s: {len(final)} objects', flush=True)
# fragment merge by joint fit (not by shared members): two objects become one only if their union fits one orbit
t0 = time.time(); merged = True
while merged and len(final) > 1:
    merged = False
    for i in range(len(final)):
        for j in range(i + 1, len(final)):
            a, b = final[i], final[j]; mem = np.union1d(a[0], b[0])
            kept, rms, c2, x, te = fit2.robust_fit(tr, mem, [(a[1], a[2]), (b[1], b[2])], thr_arcsec=max(3.0 * max(a[3], b[3]), 2.0))
            if np.isfinite(rms) and rms < 10.0 and c2 < 100.0 and len(kept) >= 0.9 * len(mem):
                final[i] = [kept, x, te, rms, c2]; del final[j]; merged = True; break
        if merged: break
print(f'fragment merge {time.time()-t0:.1f}s: {len(final)} objects', flush=True)
for kept, x, te, rms, c2 in final:
    v2 = x[3:] @ x[3:]; r = np.linalg.norm(x[:3]); eps = 0.5 * v2 - fit2.GM / r
    vinf = np.sqrt(2 * eps) * 1731.456829 if eps > 0 else float('nan')
    print(f'  object: {len(kept)} tracklets, ISO members {isiso[kept].sum()}/{isiso.sum()}, rms {rms:.2f}", chi2/dof {c2:.2f}, vinf {vinf:.1f} km/s, t span {tr["t"][kept].min():.0f}-{tr["t"][kept].max():.0f}')
