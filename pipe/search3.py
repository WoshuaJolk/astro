"""Stage 2: compiled pair/cluster stage on kernel4 output.
Each solution = (cell, tracklet, E=(vinf,bx,by,bz), tp, A=dE/d(delta), sigma). Two tracklets link if a common radiant
offset delta (within both cells) makes their elements agree within errors (4 eqs, 2 unknowns -> chi2, 2 dof) and tp agrees."""
import numpy as np, math, time, sys
from numba import njit, prange
from scipy.spatial import cKDTree
import kernel4 as K4
from kernel2 import tangent_basis
NOUT = K4.NOUT

@njit(cache=True)
def _permute_rows_inplace(val, o):
    """val[:] = val[o] without a second copy of val (cycle-following permutation)."""
    n = o.shape[0]; done = np.zeros(n, np.bool_); tmp = np.empty(val.shape[1], val.dtype)
    for start in range(n):
        if done[start] or o[start] == start:
            done[start] = True; continue
        tmp[:] = val[start]; i = start
        while True:
            j = o[i]; done[i] = True
            if j == start:
                val[i] = tmp; break
            val[i] = val[j]; i = j

def run_kernel(tr, g, cb, blo=0, bhi=1 << 30, nl=40, nr=56, step=1.5, vmin=8.0, vmax=80.0, qmax=6.0, maxc=6000, chunk=2048, keep_cells=None):
    """kernel over all tracklets, keeping only solutions whose radiant cell lies in sky bands [blo, bhi). Sorted by cell."""
    lr, rd = K4.default_scan(nl=nl, nr=nr)
    n = len(tr['t']); bhi = min(bhi, g['nb'])
    k0 = g['starts'][blo]; k1 = g['starts'][bhi] if bhi < g['nb'] else g['ncell']
    cells = []; tids = []; vals = []
    for s in range(0, n, chunk):
        ii = np.arange(s, min(n, s + chunk))
        c, rc, rv, st = K4.process_chunk(tr['t'][ii], tr['u'][ii], tr['udot'][ii], tr['obs_pos'][ii], tr['obs_vel'][ii], tr['sig_rate'][ii],
                                         lr, rd, g['nb'], g['nras'], g['starts'], g['centers'], cb, g['cell_rad'], vmin, vmax, qmax, step, maxc, 0, blo, bhi)
        tot = int(c.sum()); cc = np.empty(tot, np.int32); tt = np.empty(tot, np.int32); vv = np.empty((tot, NOUT), np.float32); p = 0
        for a in range(len(ii)):
            k = c[a]
            cc[p:p + k] = rc[a, :k]; tt[p:p + k] = ii[a]; vv[p:p + k] = rv[a, :k * NOUT].reshape(k, NOUT); p += k
        keep = keep_cells[cc] if keep_cells is not None else (cc >= k0) & (cc < k1)   # keep_cells: bool mask (ncell,) of cells a pass actually reads
        cells.append(cc[keep]); tids.append(tt[keep]); vals.append(vv[keep])
    # memory-lean assembly: fill preallocated arrays chunk by chunk (no 2x concat copy), then permute val in place
    tot = sum(len(c) for c in cells)
    cell = np.empty(tot, np.int32); tid = np.empty(tot, np.int32); val = np.empty((tot, NOUT), np.float32); p = 0
    while cells:
        c = cells.pop(0); t_ = tids.pop(0); v = vals.pop(0); k = len(c)
        cell[p:p + k] = c; tid[p:p + k] = t_; val[p:p + k] = v; p += k
        del c, t_, v
    o = np.argsort(cell, kind='stable')
    cell = cell[o]; tid = tid[o]; _permute_rows_inplace(val, o.astype(np.int64))
    return cell, tid, val

def neighbours(g):
    tree = cKDTree(g['centers']); r = 2 * math.sin(0.5 * 1.6 * g['cell_rad'] * 1.5)
    lst = tree.query_ball_point(g['centers'], r)
    maxn = max(len(l) for l in lst)
    nb_ids = np.full((g['ncell'], maxn), -1, np.int32); nb_off = np.zeros((g['ncell'], maxn, 2))
    tb = np.empty(6)
    for k, l in enumerate(lst):
        a = 0
        for kp in l:
            if kp == k: continue
            tangent_basis(g['centers'][kp], tb)
            d = g['centers'][k] - g['centers'][kp]
            nb_ids[k, a] = kp; nb_off[k, a, 0] = d @ tb[:3]; nb_off[k, a, 1] = d @ tb[3:]; a += 1
    return nb_ids, nb_off

@njit(cache=True, fastmath=True, error_model='numpy')
def pair_cell(k, cell_start, tid, val, nb_ids, nb_off, cr, chi_thr, tp_tol, narrow_v, narrow_b, out_i, out_j, out_f, cap):
    """all compatible pairs among solutions in cell k and its neighbours; delta measured from k's centre."""
    # gather
    m = cell_start[k + 1] - cell_start[k]
    for a in range(nb_ids.shape[1]):
        kp = nb_ids[k, a]
        if kp < 0: break
        m += cell_start[kp + 1] - cell_start[kp]
    if m < 2: return 0, 0
    idx = np.empty(m, np.int64); dlo = np.empty((m, 2)); dhi = np.empty((m, 2)); lo = np.empty(m); hi = np.empty(m)
    narrow = np.zeros(m, np.uint8)
    p = 0
    for a in range(-1, nb_ids.shape[1]):
        if a < 0:
            kp = k; o1 = 0.0; o2 = 0.0
        else:
            kp = nb_ids[k, a]
            if kp < 0: break
            o1 = nb_off[k, a, 0]; o2 = nb_off[k, a, 1]
        for s in range(cell_start[kp], cell_start[kp + 1]):
            idx[p] = s
            # own cell extent, expressed as delta from k's centre: own centre is at -off
            dlo[p, 0] = -o1 - 1.25 * cr; dhi[p, 0] = -o1 + 1.25 * cr; dlo[p, 1] = -o2 - 1.25 * cr; dhi[p, 1] = -o2 + 1.25 * cr
            hw = (abs(val[s, 9]) + abs(val[s, 10])) * 1.25 * cr + 3.0 * val[s, 17]
            lo[p] = val[s, 0] - hw; hi[p] = val[s, 0] + hw
            if val[s, 17] < narrow_v and val[s, 18] < narrow_b and val[s, 19] < narrow_b and val[s, 20] < narrow_b: narrow[p] = 1
            p += 1
    order = np.argsort(lo)
    n = 0; nover = 0
    # sweep on vinf interval
    for ia in range(m):
        a = order[ia]; sa = idx[a]; ta = tid[sa]
        for ib in range(ia + 1, m):
            b = order[ib]
            if lo[b] > hi[a]: break
            if narrow[a] == 0 and narrow[b] == 0: continue
            sb = idx[b]
            if tid[sb] == ta: continue
            # delta box intersection
            x0 = max(dlo[a, 0], dlo[b, 0]); x1 = min(dhi[a, 0], dhi[b, 0]); y0 = max(dlo[a, 1], dlo[b, 1]); y1 = min(dhi[a, 1], dhi[b, 1])
            if x0 > x1 or y0 > y1: continue
            # weighted LS for delta: r_c = dE_c + dA_c . delta
            h00 = 0.0; h01 = 0.0; h11 = 0.0; g0 = 0.0; g1 = 0.0
            for c in range(4):
                w = 1.0 / (val[sa, 17 + c] ** 2 + val[sb, 17 + c] ** 2 + 1e-12)
                de = val[sa, c] - val[sb, c]
                a0 = val[sa, 9 + 2 * c] - val[sb, 9 + 2 * c]; a1 = val[sa, 10 + 2 * c] - val[sb, 10 + 2 * c]
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
            chi = 0.0
            for c in range(4):
                w = 1.0 / (val[sa, 17 + c] ** 2 + val[sb, 17 + c] ** 2 + 1e-12)
                r = (val[sa, c] - val[sb, c]) + (val[sa, 9 + 2 * c] - val[sb, 9 + 2 * c]) * d0 + (val[sa, 10 + 2 * c] - val[sb, 10 + 2 * c]) * d1
                chi += w * r * r
            if chi > chi_thr: continue
            dtp = abs(val[sa, 4] - val[sb, 4])
            if dtp > tp_tol + 3.0 * math.sqrt(val[sa, 21] ** 2 + val[sb, 21] ** 2): continue
            if n >= cap:
                nover += 1; continue
            out_i[n] = sa; out_j[n] = sb; out_f[n, 0] = chi; out_f[n, 1] = d0; out_f[n, 2] = d1; out_f[n, 3] = dtp
            n += 1
    return n, nover

@njit(parallel=True, cache=True, error_model='numpy')
def pair_cells(k0, k1, cell_start, tid, val, nb_ids, nb_off, cr, chi_thr, tp_tol, narrow_v, narrow_b, cap):
    nk = k1 - k0
    out_i = np.empty((nk, cap), np.int64); out_j = np.empty((nk, cap), np.int64); out_f = np.empty((nk, cap, 4), np.float32)
    cnt = np.zeros(nk, np.int64); over = np.zeros(nk, np.int64)
    for a in prange(nk):
        n, nov = pair_cell(k0 + a, cell_start, tid, val, nb_ids, nb_off, cr, chi_thr, tp_tol, narrow_v, narrow_b, out_i[a], out_j[a], out_f[a], cap)
        cnt[a] = n; over[a] = nov
    return cnt, over, out_i, out_j, out_f

def find_pairs(cell, tid, val, g, nb_ids, nb_off, kfrom=0, kto=None, chi_thr=16.0, tp_tol=10.0, narrow_v=5.0, narrow_b=0.2, cap=20000, kchunk=512):
    cell_start = np.searchsorted(cell, np.arange(g['ncell'] + 1)).astype(np.int64)
    if kto is None: kto = g['ncell']
    P = []; tot_over = 0
    for k0 in range(kfrom, kto, kchunk):
        k1 = min(kto, k0 + kchunk)
        cnt, over, oi, oj, of = pair_cells(k0, k1, cell_start, tid, val, nb_ids, nb_off, g['cell_rad'], chi_thr, tp_tol, narrow_v, narrow_b, cap)
        tot_over += over.sum()
        for a in range(k1 - k0):
            n = cnt[a]
            if n: P.append(np.column_stack([tid[oi[a, :n]], tid[oj[a, :n]], np.full(n, k0 + a), of[a, :n, 0], of[a, :n, 3]]))
        print(f'  cells {k0}-{k1}: pairs {int(cnt.sum())} overflow {int(over.sum())}', flush=True)
    P = np.concatenate(P) if P else np.zeros((0, 5))
    return P, tot_over

def dedupe(P):
    a = np.minimum(P[:, 0], P[:, 1]).astype(np.int64); b = np.maximum(P[:, 0], P[:, 1]).astype(np.int64)
    key = a * (1 << 32) + b
    o = np.lexsort((P[:, 3], key)); key = key[o]; P = P[o]
    first = np.r_[True, key[1:] != key[:-1]]
    return P[first]

def components(P, n):
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in P[:, :2].astype(np.int64):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    return np.array([find(x) for x in range(n)])

def search(tr, g, cb, npass=4, **kw):
    """stream over sky-band passes: kernel restricted to bands [B0-1, B1+1), pairs for cells in bands [B0, B1)."""
    nb_ids, nb_off = neighbours(g)
    edges = np.linspace(0, g['nb'], npass + 1).astype(int)
    allP = []; over = 0; nsol = 0; narrow_frac = []
    for p in range(npass):
        B0, B1 = edges[p], edges[p + 1]
        t0 = time.time()
        cell, tid, val = run_kernel(tr, g, cb, max(B0 - 1, 0), min(B1 + 1, g['nb']))
        nsol += len(cell); narrow_frac.append(((val[:, 17] < 5.0) & (val[:, 18:21].max(1) < 0.2)).mean())
        t1 = time.time()
        kfrom = g['starts'][B0]; kto = g['starts'][B1] if B1 < g['nb'] else g['ncell']
        P, ov = find_pairs(cell, tid, val, g, nb_ids, nb_off, kfrom, kto, **kw)
        allP.append(P); over += ov
        print(f'pass {p}: bands {B0}-{B1}, {len(cell)} sols, kernel {t1-t0:.0f}s, pairs {len(P)} ({time.time()-t1:.0f}s)', flush=True)
        del cell, tid, val
    P = np.concatenate(allP)
    return P, over, nsol, float(np.mean(narrow_frac))
