"""Step 3b v2: cluster + screen-fit one ITF window's pairs with seed-and-grow Stage-3 (validated: 1I 52/52, 2I 624/644, 3I 2072/2143).
Per group of per-cell clusters: fit each large cluster alone (seed), grow accepted orbits over leftover tracklets (group, then whole
window) with time-expanding rounds, dedupe, then merge fragments only if their union fits one orbit. Thresholds unchanged:
rms<10", chi2/dof<100, >=3 kept. Usage: itf_fit2.py <window> [<window> ...]"""
import numpy as np, sys, time, os, pickle, glob
from collections import Counter
from search import load_tracklets
import kernel4 as K4, search6b as S4, fit2
OUT = os.environ.get('ISO_OUT', '/home/ubuntu/iso/data/itf')
tr = load_tracklets(os.environ.get('ISO_TRK', '/home/ubuntu/iso/data/itf_tracklets.npz'))
g = K4.make_grid(2.5); cb = K4.cell_bases(g); tmp = np.empty(6)
W, S = 730.0, 365.0; starts = np.arange(np.floor(tr['t'].min()), tr['t'].max() - W / 2, S)   # same windows as itf_full.py

def grow(kept, x, te, pool, rms, c2):
    for _ in range(30):
        pool = np.setdiff1d(pool, kept)
        if len(pool) == 0: break
        tmin, tmax = tr['t'][kept].min(), tr['t'][kept].max(); span = max(tmax - tmin, 10.0)
        sel = pool[(tr['t'][pool] > tmin - span) & (tr['t'][pool] < tmax + span)]
        if len(sel) == 0: break
        res = np.empty((len(sel), 4))
        if fit2.residuals(x, te, tr['t'][sel], tr['u'][sel], tr['udot'][sel], tr['obs_pos'][sel], tr['obs_vel'][sel], res, tmp) == 0: break
        add = sel[np.hypot(res[:, 0], res[:, 1]) * 206264.806 < 30.0]
        if len(add) == 0: break
        k2, rms2, c22, x2, te2 = fit2.robust_fit(tr, np.union1d(kept, add), [(x, te)], thr_arcsec=max(3.0 * rms, 2.0))
        if not (np.isfinite(rms2) and rms2 < 10.0 and c22 < 100.0 and len(np.intersect1d(kept, k2)) >= 0.9 * len(kept) and len(k2) > len(kept)): break
        kept, x, te, rms, c2 = k2, x2, te2, rms2, c22
    return kept, x, te, rms, c2

for wi in [int(a) for a in sys.argv[1:]]:
    f = f'{OUT}/pairs_w{wi:02d}.npz'
    if not os.path.exists(f): print(f'window {wi}: no pairs file'); continue
    d = np.load(f); t0 = time.time()
    if 'stream' in d.files:   # per-pass clusters written by itf_full.py (ISO_STREAM=1); identical to clustering the whole window
        cl = []; n3 = 0
        for cf in sorted(glob.glob(f'{OUT}/clusters_w{wi:02d}_p*.pkl')):
            with open(cf, 'rb') as fh: cl += pickle.load(fh)
        assert len(glob.glob(f'{OUT}/clusters_w{wi:02d}_p*.pkl')) == int(d['stream']), 'missing per-pass cluster files'
        P3 = np.zeros((0, 7))
    else:
        P3 = d['P3'].astype(np.float64); cl = S4.triangle_clusters(P3, g['cell_rad'])
    ttri = time.time() - t0
    owner = {}
    for i, c in enumerate(cl):
        for m in c[1]: owner.setdefault(int(m), []).append(i)
    cnt = Counter()
    for m, lst in owner.items():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)): cnt[(lst[a], lst[b])] += 1
    E = np.array([e for e, k in cnt.items() if k >= 2]) if cnt else np.zeros((0, 2), int)
    ccomp = S4.components(E if len(E) else np.zeros((0, 2)), len(cl)); groups = {}
    for i, c in enumerate(cl): groups.setdefault(ccomp[i], []).append(i)
    lo = starts[wi]; ALL = np.where((tr['t'] >= lo) & (tr['t'] < lo + W))[0]
    t0 = time.time(); acc = []; nfit = 0; done = np.zeros(len(tr['t']), bool)
    for gid, cis in groups.items():
        gmem = np.unique(np.concatenate([cl[i][1] for i in cis])).astype(int)
        for ci in sorted(cis, key=lambda i: -len(cl[i][1]))[:12]:
            cmem = np.asarray(cl[ci][1], int)
            if done[cmem].sum() >= 0.5 * len(cmem): continue
            sub = {k: tr[k][cmem] for k in tr}
            cell, tid, val = S4.run_kernel(sub, g, cb, 0, g['nb'])
            m = np.where(cell == cl[ci][0])[0]
            if len(m) == 0: continue
            seeds = []; used = set()
            for j in m[np.argsort(val[m, 17])]:
                if tid[j] in used: continue
                used.add(tid[j]); seeds.append(fit2.initial_state(tr, cmem[tid[j]], val[j, 7], val[j, 8]))
                if len(seeds) == 3: break
            kept, rms, c2, x, te = fit2.robust_fit(tr, cmem, seeds); nfit += 1
            if not (np.isfinite(rms) and rms < 10.0 and c2 < 100.0 and len(kept) >= 3): continue
            kept, x, te, rms, c2 = grow(kept, x, te, gmem, rms, c2)
            done[kept] = True; acc.append([kept, x, te, rms, c2])
    for a in acc: a[0], a[1], a[2], a[3], a[4] = grow(a[0], a[1], a[2], ALL, a[3], a[4])
    acc.sort(key=lambda a: (-len(a[0]), a[3])); final = []
    for a in acc:
        if any(len(np.intersect1d(a[0], q[0])) >= 0.5 * min(len(a[0]), len(q[0])) for q in final): continue
        final.append(a)
    nded = len(final); merged = True
    while merged and len(final) > 1:
        merged = False
        for i in range(len(final)):
            for j in range(i + 1, len(final)):
                a, b = final[i], final[j]; mem = np.union1d(a[0], b[0])
                kept, rms, c2, x, te = fit2.robust_fit(tr, mem, [(a[1], a[2]), (b[1], b[2])], thr_arcsec=max(3.0 * max(a[3], b[3]), 2.0))
                if np.isfinite(rms) and rms < 10.0 and c2 < 100.0 and len(kept) >= 0.9 * len(mem):
                    final[i] = [kept, x, te, rms, c2]; del final[j]; merged = True; break
            if merged: break
    print(f'window {wi:02d} v2: {len(P3)} chi<3 pairs, {len(cl)} per-cell clusters ({ttri:.0f}s), {len(groups)} groups, {nfit} seed fits, '
          f'{len(acc)} accepted seeds -> {nded} after grow/dedupe -> {len(final)} objects, fits {time.time()-t0:.0f}s', flush=True)
    for kept, x, te, rms, c2 in sorted(final, key=lambda a: (-len(a[0]), a[3])):
        sp = sorted(set(tr['station'][kept].astype(str))); tt = tr['t'][kept]
        v2 = x[3:] @ x[3:]; r = np.linalg.norm(x[:3]); eps = 0.5 * v2 - fit2.GM / r
        vinf = np.sqrt(2 * eps) * 1731.456829 if eps > 0 else float('nan')
        print(f'   CANDIDATE2: {len(kept)} tracklets, rms {rms:.2f}", chi2/dof {c2:.1f}, vinf {vinf:.1f} km/s, MJD {tt.min():.1f}-{tt.max():.1f} ({tt.max()-tt.min():.0f} d), stations {sp}, desigs {sorted(set(tr["desig"][kept].astype(str)))[:8]}', flush=True)
    np.save(f'{OUT}/fit2_w{wi:02d}.npy', np.array([(k.tolist(), x.tolist(), te, rms, c2) for k, x, te, rms, c2 in final], dtype=object), allow_pickle=True)
