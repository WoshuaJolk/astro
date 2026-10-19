"""Window-based hyperbolic linking of tracklets: kernel -> per-cell clustering -> candidate pairs."""
import numpy as np, time, sys, math
from scipy.spatial import cKDTree
from kernel2 import *

STATION_SIG = {b'F51': 0.12, b'F52': 0.12, b'W84': 0.10, b'X05': 0.10, b'T09': 0.25, b'T05': 0.25, b'T08': 0.25,
               b'G96': 0.35, b'703': 0.6, b'I41': 0.3, b'C51': 0.5, b'V00': 0.3, b'691': 0.4, b'568': 0.15, b'309': 0.15, b'807': 0.2}

def load_tracklets(path, rate_max_deg=12.0, rms_max=2.0):
    d = np.load(path)
    rate = np.rad2deg(np.linalg.norm(d['udot'], axis=1))
    sel = (d['nobs'] >= 2) & (rate < rate_max_deg) & (d['rms_arcsec'] < rms_max) & (d['arc_days'] > 2.0 / 1440) & np.isfinite(d['t'])
    tr = {k: d[k][sel] for k in ['t', 'u', 'udot', 'obs_pos', 'obs_vel', 'nobs', 'arc_days', 'rms_arcsec', 'mag', 'desig', 'station', 'first_line']}
    sig_pos = np.array([STATION_SIG.get(s, 0.4) for s in tr['station']])
    sig_pos = np.maximum(sig_pos, tr['rms_arcsec'])  # arcsec
    # slope error of linear fit ~ sig / sqrt(sum dt^2), sum dt^2 ~ n*arc^2/12
    stt = tr['nobs'] * tr['arc_days'] ** 2 / 12.0
    tr['sig_rate'] = np.deg2rad(sig_pos / 3600.0) / np.sqrt(stt)  # rad/day
    tr['rate'] = np.deg2rad(rate[sel])
    print(f'loaded {len(tr["t"])} tracklets ({sel.sum()}/{len(sel)} kept); t range {tr["t"].min():.0f}-{tr["t"].max():.0f}', flush=True)
    return tr

def run_kernel(tr, idx, g, lr, rd, vmin=8.0, vmax=80.0, qmax=6.0, maxc=2500, chunk=4096):
    cells = []; vals = []; tids = []
    t0 = time.time(); tot = 0
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        c, rc, rv = process_chunk(tr['t'][ii], tr['u'][ii], tr['udot'][ii], tr['obs_pos'][ii], tr['obs_vel'][ii], tr['sig_rate'][ii],
                                  lr, rd, g['nb'], g['nras'], g['starts'], g['centers'], g['cell_rad'], vmin, vmax, qmax, maxc)
        for a in range(len(ii)):
            n = c[a]
            if n == 0: continue
            cells.append(rc[a, :n].copy()); vals.append(rv[a, :n * NOUT].reshape(n, NOUT).astype(np.float32).copy() if False else rv[a, :n * NOUT].reshape(n, NOUT).copy())
            tids.append(np.full(n, ii[a], np.int64))
        tot += c.sum()
    cells = np.concatenate(cells); vals = np.concatenate(vals); tids = np.concatenate(tids)
    print(f'  kernel: {len(idx)} tracklets -> {tot} cell-solutions in {time.time()-t0:.1f}s', flush=True)
    return tids, cells, vals

def cluster(tr, tids, cells, vals, g, sv=2.0, sb=0.15, st=6.0, chi_max=4.0, dt_min=0.3, dt_max=400.0, floor=(0.3, 0.01, 0.5)):
    """find pairs of different tracklets whose (vinf, b, tp) agree within errors, per cell"""
    t0 = time.time()
    order = np.argsort(cells, kind='stable'); cells = cells[order]; tids = tids[order]; vals = vals[order]
    bounds = np.searchsorted(cells, np.arange(g['ncell'] + 1))
    pairs = []
    T = tr['t']
    for k in range(g['ncell']):
        a, b = bounds[k], bounds[k + 1]
        if b - a < 2: continue
        V = vals[a:b]; ti = tids[a:b]
        X = np.column_stack([V[:, 0] / sv, V[:, 1] / sb, V[:, 2] / sb, V[:, 3] / sb, V[:, 4] / st])
        tree = cKDTree(X)
        pr = tree.query_pairs(1.0, output_type='ndarray')
        if len(pr) == 0: continue
        i, j = pr[:, 0], pr[:, 1]
        ok = ti[i] != ti[j]
        dt = np.abs(T[ti[i]] - T[ti[j]])
        ok &= (dt > dt_min) & (dt < dt_max)
        if not ok.any(): continue
        i, j = i[ok], j[ok]
        # normalised distance using per-point sigmas
        s_v2 = V[i, 5] ** 2 + V[j, 5] ** 2 + floor[0] ** 2
        s_b2 = V[i, 6] ** 2 + V[j, 6] ** 2 + floor[1] ** 2
        s_t2 = V[i, 7] ** 2 + V[j, 7] ** 2 + floor[2] ** 2
        chi2 = (V[i, 0] - V[j, 0]) ** 2 / s_v2 + ((V[i, 1] - V[j, 1]) ** 2 + (V[i, 2] - V[j, 2]) ** 2 + (V[i, 3] - V[j, 3]) ** 2) / s_b2 + (V[i, 4] - V[j, 4]) ** 2 / s_t2
        ok = chi2 < chi_max ** 2
        if not ok.any(): continue
        i, j = i[ok], j[ok]
        pairs.append(np.column_stack([np.minimum(ti[i], ti[j]), np.maximum(ti[i], ti[j]), np.full(ok.sum(), k), chi2[ok], V[i, 0], V[i, 10], V[i, 8], V[j, 8]]))
    if not pairs:
        print('  no pairs'); return np.zeros((0, 8))
    P = np.concatenate(pairs)
    # unique pairs, keep best chi2
    key = P[:, 0].astype(np.int64) * 10_000_000 + P[:, 1].astype(np.int64)
    o = np.lexsort((P[:, 3], key)); P = P[o]; key = key[o]
    first = np.ones(len(key), bool); first[1:] = key[1:] != key[:-1]
    ncell_hits = np.diff(np.append(np.where(first)[0], len(key)))
    P = P[first]
    P = np.column_stack([P, ncell_hits])
    print(f'  cluster: {len(P)} unique pairs in {time.time()-t0:.1f}s', flush=True)
    return P

def search(path, cell_deg=2.5, win=730.0, stride=365.0, tmin=None, tmax=None, **kw):
    tr = load_tracklets(path)
    g = make_grid(cell_deg); lr, rd = default_scan()
    t = tr['t']
    t0 = tmin if tmin is not None else np.floor(t.min()); t1 = tmax if tmax is not None else t.max()
    allP = []
    w0 = t0
    while w0 < t1:
        idx = np.where((t >= w0) & (t < w0 + win))[0]
        if len(idx) >= 2:
            print(f'window {w0:.0f}-{w0+win:.0f}: {len(idx)} tracklets', flush=True)
            tids, cells, vals = run_kernel(tr, idx, g, lr, rd)
            P = cluster(tr, tids, cells, vals, g, **kw)
            if len(P): allP.append(P)
        w0 += stride
    P = np.concatenate(allP) if allP else np.zeros((0, 9))
    return tr, P

if __name__ == '__main__':
    tr, P = search(sys.argv[1])
    np.save(sys.argv[2], P)
