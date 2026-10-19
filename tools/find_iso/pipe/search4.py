"""Stage 2 v2: hashed sheet linking. Each solution is a linear sheet E(delta)=E0+A.delta over its radiant cell. We evaluate
the sheet at NSUB x NSUB sub-cell centres (cheap, from A), bin (vinf,b,tp) with bin = 2*tol and insert into 32 shifted grids
(any two points within tol in all dims share a key). Equal keys from different tracklets -> exact chi2 pair test (search3)."""
import numpy as np, math, time
from numba import njit, prange
import search3 as S3
from search3 import run_kernel, neighbours, dedupe, components

TOL = np.array([0.5, 0.017, 0.017, 0.017, 0.5])          # km/s, AU x3, days  (half-tolerance; bin = 2*tol)
OFFS = np.array([0.0, -40.0, -8.0, -8.0, -8.0, 0.0])  # unused placeholder

@njit(cache=True, fastmath=True, error_model='numpy')
def exact_pair(sa, sb, val, x0, x1, y0, y1, chi_thr, tp_tol):
    h00 = 0.0; h01 = 0.0; h11 = 0.0; g0 = 0.0; g1 = 0.0
    for c in range(5):
        oi = 9 + 2 * c if c < 4 else 22
        w = 1.0 / (val[sa, 17 + c] ** 2 + val[sb, 17 + c] ** 2 + 1e-12)
        de = val[sa, c] - val[sb, c]
        a0 = val[sa, oi] - val[sb, oi]; a1 = val[sa, oi + 1] - val[sb, oi + 1]
        h00 += w * a0 * a0; h01 += w * a0 * a1; h11 += w * a1 * a1; g0 += w * a0 * de; g1 += w * a1 * de
    det = h00 * h11 - h01 * h01
    if det > 1e-20:
        d0 = -(h11 * g0 - h01 * g1) / det; d1 = -(h00 * g1 - h01 * g0) / det
    else:
        d0 = 0.5 * (x0 + x1); d1 = 0.5 * (y0 + y1)
    if d0 < x0: d0 = x0
    if d0 > x1: d0 = x1
    if d1 < y0: d1 = y0
    if d1 > y1: d1 = y1
    chi = 0.0; dtp = 0.0
    for c in range(5):
        oi = 9 + 2 * c if c < 4 else 22
        w = 1.0 / (val[sa, 17 + c] ** 2 + val[sb, 17 + c] ** 2 + 1e-12)
        r = (val[sa, c] - val[sb, c]) + (val[sa, oi] - val[sb, oi]) * d0 + (val[sa, oi + 1] - val[sb, oi + 1]) * d1
        chi += w * r * r
        if c == 4: dtp = abs(r)
    if chi > chi_thr: return -1.0, d0, d1, dtp
    return chi, d0, d1, dtp

NSUB = 8
NV = 80; NB = 320; NT = 1024; HBITS = 22   # NV: v_inf hash bins of 2*TOL[0]=1 km/s; was 40 -> solutions with v_inf >= 40 km/s got no hash key (fixed 2026-10-19)

@njit(cache=True, fastmath=True, error_model='numpy')
def hash_cell(k, cell_start, tid, val, T, narrow, nb_ids, nb_off, cr, tol, tp0, chi_thr, tp_tol, dt_min, out_i, out_j, out_f, cap, maxrun, counts, table):
    """counting-sort bucket hash on (sub-cell, vinf bin, bx bin) with 2x2 shifted grids; by/bz/tp checked in-bucket."""
    m = cell_start[k + 1] - cell_start[k]
    for a in range(nb_ids.shape[1]):
        kp = nb_ids[k, a]
        if kp < 0: break
        m += cell_start[kp + 1] - cell_start[kp]
    if m < 2: return 0, 0, 0
    idx = np.empty(m, np.int64); off = np.empty((m, 2))
    p = 0
    for a in range(-1, nb_ids.shape[1]):
        if a < 0:
            kp = k; o1 = 0.0; o2 = 0.0
        else:
            kp = nb_ids[k, a]
            if kp < 0: break
            o1 = nb_off[k, a, 0]; o2 = nb_off[k, a, 1]
        for s in range(cell_start[kp], cell_start[kp + 1]):
            if narrow[s] == 0: continue
            idx[p] = s; off[p, 0] = o1; off[p, 1] = o2; p += 1
    m = p
    if m < 2: return 0, 0, 0
    ext = 1.25 * cr
    nbk = NSUB * NSUB * 4 * NV * NT
    counts[:nbk + 1] = 0
    tv = 2 * tol[0]; tt = 2 * tol[4]
    vb0 = 0.0  # vinf bins start at 0 km/s
    # pass 1: count
    for p in range(m):
        s = idx[p]
        for jx in range(NSUB):
            dx = (jx + 0.5) / NSUB * 2 * cr - cr
            ddx = dx + off[p, 0]
            if abs(ddx) > ext: continue
            for jy in range(NSUB):
                dy = (jy + 0.5) / NSUB * 2 * cr - cr
                ddy = dy + off[p, 1]
                if abs(ddy) > ext: continue
                j = jx * NSUB + jy
                v = val[s, 0] + val[s, 9] * ddx + val[s, 10] * ddy
                tpv = val[s, 4] + val[s, 22] * ddx + val[s, 23] * ddy - tp0
                for sh in range(4):
                    bv = int(math.floor(v / tv + 0.5 * (sh & 1))); bb = int(math.floor(tpv / tt + 0.5 * (sh >> 1))) + NT // 2
                    if bv < 0 or bv >= NV or bb < 0 or bb >= NT: continue
                    counts[((j * 4 + sh) * NV + bv) * NT + bb + 1] += 1
    for q in range(nbk): counts[q + 1] += counts[q]
    tot = counts[nbk]
    who = np.empty(tot, np.int32); pos = counts[:nbk].copy()
    for p in range(m):
        s = idx[p]
        for jx in range(NSUB):
            dx = (jx + 0.5) / NSUB * 2 * cr - cr
            ddx = dx + off[p, 0]
            if abs(ddx) > ext: continue
            for jy in range(NSUB):
                dy = (jy + 0.5) / NSUB * 2 * cr - cr
                ddy = dy + off[p, 1]
                if abs(ddy) > ext: continue
                j = jx * NSUB + jy
                v = val[s, 0] + val[s, 9] * ddx + val[s, 10] * ddy
                tpv = val[s, 4] + val[s, 22] * ddx + val[s, 23] * ddy - tp0
                for sh in range(4):
                    bv = int(math.floor(v / tv + 0.5 * (sh & 1))); bb = int(math.floor(tpv / tt + 0.5 * (sh >> 1))) + NT // 2
                    if bv < 0 or bv >= NV or bb < 0 or bb >= NT: continue
                    q = ((j * 4 + sh) * NV + bv) * NT + bb
                    who[pos[q]] = p; pos[q] += 1
    # pass 3: pairs within buckets; dedupe tested pairs with open-addressing set
    table[:] = -1
    hmask = (1 << HBITS) - 1
    n = 0; nover = 0; nrunover = 0
    x0 = -ext; x1 = ext
    for q in range(nbk):
        a = counts[q]; b = counts[q + 1]; L = b - a
        if L < 2: continue
        if L > maxrun: nrunover += 1
        j = q // (4 * NV * NT)
        jx = j // NSUB; jy = j % NSUB
        dx = (jx + 0.5) / NSUB * 2 * cr - cr; dy = (jy + 0.5) / NSUB * 2 * cr - cr
        # lossless handling of long runs: sort the run by bx and sweep (no truncation)
        bxs = np.empty(L); bys = np.empty(L); bzs = np.empty(L)
        for ii in range(L):
            pi = who[a + ii]; si = idx[pi]
            ddx = dx + off[pi, 0]; ddy = dy + off[pi, 1]
            bxs[ii] = val[si, 1] + val[si, 11] * ddx + val[si, 12] * ddy
            bys[ii] = val[si, 2] + val[si, 13] * ddx + val[si, 14] * ddy
            bzs[ii] = val[si, 3] + val[si, 15] * ddx + val[si, 16] * ddy
        order = np.argsort(bxs)
        for oi in range(L):
            ii = order[oi]; pi = who[a + ii]; si = idx[pi]; ti = tid[si]
            for oj in range(oi + 1, L):
                jj = order[oj]
                if bxs[jj] - bxs[ii] > tol[1]: break
                pj = who[a + jj]; sj = idx[pj]; tj = tid[sj]
                if tj == ti or abs(T[ti] - T[tj]) < dt_min: continue
                if abs(bys[ii] - bys[jj]) > tol[2] or abs(bzs[ii] - bzs[jj]) > tol[3]: continue
                # dedupe
                lo_ = ti if ti < tj else tj; hi_ = tj if ti < tj else ti
                key = lo_ * 1000003 + hi_
                h = ((key ^ (key >> 31)) * 2654435761) & hmask
                seen = False
                while table[h] != -1:
                    if table[h] == key: seen = True; break
                    h = (h + 1) & hmask
                if seen: continue
                table[h] = key
                chi, d0, d1, dtp = exact_pair(si, sj, val, x0, x1, x0, x1, chi_thr, tp_tol)
                if chi < 0: continue
                if n >= cap: nover += 1; continue
                out_i[n] = si; out_j[n] = sj; out_f[n, 0] = chi; out_f[n, 1] = d0; out_f[n, 2] = d1; out_f[n, 3] = dtp; n += 1
    return n, nover, nrunover

@njit(parallel=True, cache=True, error_model='numpy')
def hash_cells(k0, k1, cell_start, tid, val, T, narrow, nb_ids, nb_off, cr, tol, tp0, chi_thr, tp_tol, dt_min, cap, maxrun):
    nk = k1 - k0
    out_i = np.empty((nk, cap), np.int64); out_j = np.empty((nk, cap), np.int64); out_f = np.empty((nk, cap, 4), np.float32)
    cnt = np.zeros(nk, np.int64); over = np.zeros(nk, np.int64); rover = np.zeros(nk, np.int64)
    nthr = 8
    counts = np.empty((nthr, NSUB * NSUB * 4 * NV * NT + 1), np.int32); table = np.empty((nthr, 1 << HBITS), np.int64)
    for th in prange(nthr):
        for a in range(th, nk, nthr):
            n, nov, nro = hash_cell(k0 + a, cell_start, tid, val, T, narrow, nb_ids, nb_off, cr, tol, tp0, chi_thr, tp_tol, dt_min, out_i[a], out_j[a], out_f[a], cap, maxrun, counts[th], table[th])
            cnt[a] = n; over[a] = nov; rover[a] = nro
    return cnt, over, rover, out_i, out_j, out_f

NARROW = (5.0, 0.2, 5.0)   # sigma_vinf km/s, sigma_b AU, sigma_tp d: 'informative solution' thresholds (error-based, not physics)

def narrow_flags(val):
    return ((val[:, 17] < NARROW[0]) & (val[:, 18:21].max(1) < NARROW[1]) & (val[:, 21] < NARROW[2])).astype(np.uint8)

def find_pairs(cell, tid, val, T, g, nb_ids, nb_off, kfrom=0, kto=None, tp0=0.0, chi_thr=20.0, tp_tol=10.0, dt_min=1.0, cap=200000, maxrun=256, kchunk=256):
    cell_start = np.searchsorted(cell, np.arange(g['ncell'] + 1)).astype(np.int64)
    narrow = narrow_flags(val)
    if kto is None: kto = g['ncell']
    P = []; tot_over = 0; tot_rover = 0
    for k0 in range(kfrom, kto, kchunk):
        k1 = min(kto, k0 + kchunk)
        cnt, over, rover, oi, oj, of = hash_cells(k0, k1, cell_start, tid, val, T, narrow, nb_ids, nb_off, g['cell_rad'], TOL, tp0, chi_thr, tp_tol, dt_min, cap, maxrun)
        tot_over += over.sum(); tot_rover += rover.sum()
        for a in range(k1 - k0):
            n = cnt[a]
            if n: P.append(np.column_stack([tid[oi[a, :n]], tid[oj[a, :n]], np.full(n, k0 + a), of[a, :n, 0], of[a, :n, 3], of[a, :n, 1], of[a, :n, 2]]))
    P = np.concatenate(P) if P else np.zeros((0, 7))
    return P, tot_over, tot_rover

def search(tr, g, cb, npass=4, **kw):
    nb_ids, nb_off = neighbours(g)
    tp0 = float(np.median(tr['t']))
    edges = np.linspace(0, g['nb'], npass + 1).astype(int)
    allP = []; over = 0; rover = 0; nsol = 0
    for p in range(npass):
        B0, B1 = edges[p], edges[p + 1]
        t0 = time.time()
        cell, tid, val = run_kernel(tr, g, cb, max(B0 - 1, 0), min(B1 + 1, g['nb']))
        nsol += len(cell); t1 = time.time()
        kfrom = g['starts'][B0]; kto = g['starts'][B1] if B1 < g['nb'] else g['ncell']
        P, ov, ro = find_pairs(cell, tid, val, tr['t'], g, nb_ids, nb_off, kfrom, kto, tp0=tp0, **kw)
        allP.append(P); over += ov; rover += ro
        print(f'pass {p}: bands {B0}-{B1}, {len(cell)} sols, kernel {t1-t0:.0f}s, pairs {len(P)} ({time.time()-t1:.0f}s) overflow {ov} runs>max {ro}', flush=True)
        del cell, tid, val
    return np.concatenate(allP), over, nsol

def triangle_clusters(P, cr, dmax=0.35, dense=50000):
    """P columns: ti, tj, cell, chi, dtp, d0, d1. Within each cell, keep only edges that belong to a triangle whose three
    radiant solutions delta* agree within dmax*cr; connected components of those edges = clusters (per cell)."""
    from collections import defaultdict
    out = []
    o = np.argsort(P[:, 2], kind='stable'); P = P[o]; cells = P[:, 2].astype(np.int64)
    bounds = np.r_[0, np.flatnonzero(np.diff(cells)) + 1, len(P)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 3: continue
        Q = P[s:e]; ti = Q[:, 0].astype(np.int64); tj = Q[:, 1].astype(np.int64)
        if e - s > dense:
            # far beyond any random pair density: accept the cell's edges without triangle enumeration
            keep = np.ones(len(Q), bool)
        else:
          keep = np.zeros(len(Q), bool)
          if True:
            adj = defaultdict(dict)
            for r in range(len(Q)):
                adj[ti[r]][tj[r]] = r; adj[tj[r]][ti[r]] = r
            for r in range(len(Q)):
                a, b = ti[r], tj[r]
                na = adj[a]; nb = adj[b]
                if len(na) > len(nb): na, nb = nb, na
                for c, r2 in na.items():
                    r3 = nb.get(c)
                    if r3 is None: continue
                    d = Q[[r, r2, r3], 5:7]
                    if np.abs(d - d.mean(0)).max() < dmax * cr: keep[r] = keep[r2] = keep[r3] = True
        if not keep.any(): continue
        Qk = Q[keep]; nodes = np.unique(np.r_[Qk[:, 0], Qk[:, 1]]).astype(np.int64); pos = {n: i for i, n in enumerate(nodes)}
        parent = list(range(len(nodes)))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for a, b in Qk[:, :2].astype(np.int64):
            ra, rb = find(pos[a]), find(pos[b])
            if ra != rb: parent[ra] = rb
        roots = np.array([find(i) for i in range(len(nodes))])
        for rt in np.unique(roots):
            mem = nodes[roots == rt]
            if len(mem) >= 3: out.append((int(cells[s]), mem, float(Qk[:, 5].mean()), float(Qk[:, 6].mean())))
    return out
