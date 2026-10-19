"""Step 3: full ITF search. 2-year windows (kernel tp range is +-512 d around the window median), 1-year stride so every pair of
tracklets <1 yr apart shares a window. Per window: kernel + hashed sheet match (search6b) in memory-bounded passes, dedupe,
save chi<3 per-cell pairs (for triangles) and all unique pairs with GLOBAL tracklet ids. Usage: itf_full.py [--plan] [first_window [last_window]]"""
import numpy as np, pickle, glob
import search6b as S6, time, sys, os
from search import load_tracklets
import kernel4 as K4, search6b as S3, search3 as SK
PASS_TRK = int(os.environ.get('ISO_PASS_TRK', '16000'))
KCACHE = os.environ.get('ISO_KCACHE', '1') == '1'; KBANDS = int(os.environ.get('ISO_KBANDS', '2'))   # per-band kernel cache (search3.run_kernel_cached)   # tracklets per pass unit; sols/pass ~ N/PASS_TRK

def dedupe_lean(P, per_cell=False):
    """Same result as search3.dedupe (keep lowest-chi row per unordered tracklet pair) without the float64 copy.
    per_cell=True keys on (pair, cell) instead: a lossless compaction that keeps the per-cell rows P3 needs."""
    if len(P) == 0: return P
    a = P[:, 0].astype(np.int64); b = P[:, 1].astype(np.int64)
    if per_cell:
        c = P[:, 2].astype(np.int64); ba = int(max(a.max(), b.max())).bit_length(); bc = int(c.max()).bit_length()
        assert 2 * ba + bc < 63
        key = (np.minimum(a, b) << (ba + bc)) + (np.maximum(a, b) << bc) + c; del c
    else: key = np.minimum(a, b) * (1 << 32) + np.maximum(a, b)
    del a, b
    o = np.lexsort((P[:, 3], key)); key = key[o]
    first = np.r_[True, key[1:] != key[:-1]]
    return P[o[first]]
OUT = os.environ.get('ISO_OUT', '/home/ubuntu/iso/data/itf')
tr = load_tracklets(os.environ.get('ISO_TRK', '/home/ubuntu/iso/data/itf_tracklets.npz'))
t = tr['t']; W = 730.0; S = 365.0
starts = np.arange(np.floor(t.min()), t.max() - W / 2, S)
plan = '--plan' in sys.argv; args = [a for a in sys.argv[1:] if not a.startswith('--')]
w0 = int(args[0]) if args else 0; w1 = int(args[1]) if len(args) > 1 else 10**9
if plan:
    for wi, ws in enumerate(starts):
        n = int(((t >= ws) & (t < ws + W)).sum()); print(f'window {wi:2d}: MJD {ws:.0f}-{ws+W:.0f}  tracklets {n:7d}  npass {max(4, int(np.ceil(n / PASS_TRK)))}')
    sys.exit()
g = K4.make_grid(2.5); cb = K4.cell_bases(g); nb_ids, nb_off = S3.neighbours(g)
STREAM = os.environ.get('ISO_STREAM', '0') == '1'   # per-pass triangle clustering; never holds all passes in RAM

def cluster_pass(P, cf):
    """Lossless w.r.t. itf_fit2: it only uses triangle-surviving edges, and a pass covers a disjoint radiant-cell range,
    so clustering per pass == clustering the concatenated window."""
    cl = S6.triangle_clusters(P[P[:, 3] < 3.0].astype(np.float64), g['cell_rad'])
    with open(cf, 'wb') as fh: pickle.dump(cl, fh)
    return len(cl)

for wi, ws in enumerate(starts):
    if wi < w0 or wi > w1: continue
    f = f'{OUT}/pairs_w{wi:02d}.npz'
    if os.path.exists(f): continue
    sel = np.flatnonzero((t >= ws) & (t < ws + W))
    if len(sel) < 10: continue
    sub = {k: v[sel] for k, v in tr.items()}
    npass = max(4, int(np.ceil(len(sel) / PASS_TRK)))
    tp0 = float(np.median(sub['t'])); cedges = np.linspace(0, g['ncell'], npass + 1).astype(int)   # cell-based: passes may be finer than a band
    tw = time.time(); allP = []; nsol = 0; over = 0; tk = 0.0; nraw = 0; ncl = 0
    PR = os.environ.get('ISO_PASS_RANGE')   # "a:b" -> only passes a..b-1 (worker split); window summary needs all passes
    plo, phi = (int(x) for x in PR.split(':')) if PR else (0, npass)
    for p in range(npass):
        if p < plo or p >= phi: continue
        kfrom, kto = int(cedges[p]), int(cedges[p + 1]); t0 = time.time()
        B0 = int(np.searchsorted(g['starts'], kfrom, 'right') - 1); B1 = int(np.searchsorted(g['starts'], kto - 1, 'right'))
        pf = f'{OUT}/pairs_w{wi:02d}_p{p:02d}.npy'
        if STREAM and not os.path.exists(pf) and os.path.exists(f'{OUT}/clusters_w{wi:02d}_p{p:02d}.pkl'):   # pairs checkpoint deleted to free disk: the pkl is the deliverable
            with open(f'{OUT}/clusters_w{wi:02d}_p{p:02d}.pkl', 'rb') as fh: ncl += len(pickle.load(fh))
            print(f'  w{wi:02d} pass {p}/{npass}: clusters checkpoint present (pairs file removed), skipped', flush=True); continue
        if os.path.exists(pf):  # per-pass checkpoint: a restart re-uses finished passes
            P = np.load(pf); nraw += len(P)
            if STREAM:
                cf = f'{OUT}/clusters_w{wi:02d}_p{p:02d}.pkl'
                if not os.path.exists(cf): ncl += cluster_pass(P, cf)
                else:
                    with open(cf, 'rb') as fh: ncl += len(pickle.load(fh))
            else: allP.append(P)
            print(f'  w{wi:02d} pass {p}/{npass}: loaded {len(P)} pairs from checkpoint', flush=True); continue
        keep_cells = np.zeros(g['ncell'], bool); keep_cells[kfrom:kto] = True
        nbk = nb_ids[kfrom:kto].ravel(); keep_cells[nbk[nbk >= 0]] = True   # hasher reads a cell's sols and its neighbour cells' sols only
        if KCACHE: cell, tid, val = SK.run_kernel_cached(sub, g, cb, max(B0 - 1, 0), min(B1 + 1, g['nb']), keep_cells, f'{OUT}/kcache', f'w{wi:02d}', nbands=KBANDS)   # kernel once per band, cached on disk
        else: cell, tid, val = S3.run_kernel(sub, g, cb, max(B0 - 1, 0), min(B1 + 1, g['nb']), keep_cells=keep_cells)
        nsol += len(cell); tk += time.time() - t0
        P, ov, ro = S3.find_pairs(cell, tid, val, sub['t'], g, nb_ids, nb_off, kfrom, kto, tp0=tp0)
        P[:, 0] = sel[P[:, 0].astype(int)]; P[:, 1] = sel[P[:, 1].astype(int)]
        P = dedupe_lean(P.astype(np.float32), per_cell=True); np.save(pf, P); over += ov; nraw += len(P)
        if STREAM:
            cf = f'{OUT}/clusters_w{wi:02d}_p{p:02d}.pkl'
            if os.path.exists(cf): os.remove(cf)   # a finished-pass marker must be newer than its pairs file
            ncl += cluster_pass(P, cf)
            if os.environ.get('ISO_KEEP_PAIRS', '0') != '1': os.remove(pf)   # the pkl is the deliverable and marks the pass done; the pairs file only costs disk
        else: allP.append(P)
        print(f'  w{wi:02d} pass {p}/{npass}: {len(cell)} sols, pairs {len(P)}, overflow {ov}, {time.time()-t0:.0f}s', flush=True)
        del cell, tid, val, P
    if STREAM:
        if PR and len(glob.glob(f'{OUT}/clusters_w{wi:02d}_p*.pkl')) < npass:
            print(f'window {wi:02d}: passes {plo}-{phi-1} done ({nraw} raw pairs, {ncl} clusters, overflow {over}); window incomplete, no summary', flush=True); continue
        np.savez(f, stream=np.int64(npass), sel=sel, mjd0=ws, mjd1=ws + W)
        print(f'window {wi:02d} MJD {ws:.0f}-{ws+W:.0f}: {len(sel)} tracklets, {nsol} solutions, {nraw} raw pairs, {ncl} per-cell clusters (streamed, {npass} passes), overflow {over}, kernel {tk:.0f}s, total {time.time()-tw:.0f}s', flush=True)
        continue
    P_all = np.concatenate(allP); del allP
    Pd = dedupe_lean(P_all)
    np.savez(f, P3=P_all[P_all[:, 3] < 3.0], Pd=Pd, sel=sel, mjd0=ws, mjd1=ws + W)
    print(f'window {wi:02d} MJD {ws:.0f}-{ws+W:.0f}: {len(sel)} tracklets, {nsol} solutions, {len(P_all)} raw pairs, {len(Pd)} unique, overflow {over}, kernel {tk:.0f}s, total {time.time()-tw:.0f}s', flush=True)
