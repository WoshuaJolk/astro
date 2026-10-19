"""Stage 2 v3: same sheet-hash linking as search4 (same TOL, NSUB, exact_pair, dedupe) but:
 - sparse bucketing: inserts are radix-sorted by key, no 10.5M dense count array per cell
 - one hash grid + neighbour-bucket probing (+1 in tp, +1 in vinf, diagonals) instead of 4 shifted grids; explicit |dv|<=tol, |dtp|<=tol
   at the sample -> criterion identical to search4's guarantee (within tol in all 5 dims at some sample point)
 - dedupe table cleared via touched list, not a 32 MB memset per cell
 - fully degenerate bucket runs (all members within tol in b) are remembered by set-hash and not re-swept at later sub-cells
Pairs found are identical to search4's (any two sheet samples within tol in all dims land in same/adjacent buckets)."""
import numpy as np, math, time
from numba import njit, prange
from search3 import run_kernel, neighbours, dedupe, components
from search4 import exact_pair, narrow_flags, TOL, NSUB, NV, NT, HBITS, triangle_clusters

NKEY = NSUB * NSUB * NV * NT
assert NKEY < (1 << 24)  # 2 radix passes x RBITS
RBITS = 12; RMASK = (1 << RBITS) - 1
TBITS = 21

@njit(cache=True, error_model='numpy')
def radix_sort(K, P, n, K2, P2, cnt):
    for sh in (0, RBITS):
        cnt[:] = 0
        for i in range(n): cnt[((K[i] >> sh) & RMASK) + 1] += 1
        for d in range(RMASK + 1): cnt[d + 1] += cnt[d]
        for i in range(n):
            d = (K[i] >> sh) & RMASK; q = cnt[d]; K2[q] = K[i]; P2[q] = P[i]; cnt[d] = q + 1
        for i in range(n): K[i] = K2[i]; P[i] = P2[i]

@njit(cache=True, fastmath=True, error_model='numpy')
def emit(si, sj, ti, tj, val, x0, x1, chi_thr, tp_tol, table, hmask, touched, st, out_i, out_j, out_f, cap, slot, slot_touched):
    """test solution pair once (table), keep per tracklet pair the min-chi row (slot map)."""
    st[5] += 1
    lo_ = si if si < sj else sj; hi_ = sj if si < sj else si
    key = lo_ * 1000000007 + hi_
    h = ((key ^ (key >> 31)) * 2654435761) & hmask
    while table[h] != -1:
        if table[h] == key: return
        h = (h + 1) & hmask
    table[h] = key
    touched[st[1]] = h; st[1] += 1
    if st[1] == touched.shape[0]:   # table half full: reset (dedupe restarts; slot map keeps output exact)
        for i in range(st[1]): table[touched[i]] = -1
        st[1] = 0; st[4] += 1
    chi, d0, d1, dtp = exact_pair(si, sj, val, x0, x1, x0, x1, chi_thr, tp_tol)
    if chi < 0: return
    lo_ = ti if ti < tj else tj; hi_ = tj if ti < tj else ti
    tkey = lo_ * 1000003 + hi_
    g = ((tkey ^ (tkey >> 29)) * 11400714819323198485) & hmask
    while slot[g, 0] != -1:
        if slot[g, 0] == tkey:
            n = slot[g, 1]
            if chi < out_f[n, 0]:
                out_i[n] = si; out_j[n] = sj; out_f[n, 0] = chi; out_f[n, 1] = d0; out_f[n, 2] = d1; out_f[n, 3] = dtp
            return
        g = (g + 1) & hmask
    n = st[0]
    if n >= cap: st[2] += 1; return
    slot[g, 0] = tkey; slot[g, 1] = n
    if st[3] < slot_touched.shape[0]: slot_touched[st[3]] = g
    st[3] += 1
    out_i[n] = si; out_j[n] = sj; out_f[n, 0] = chi; out_f[n, 1] = d0; out_f[n, 2] = d1; out_f[n, 3] = dtp; st[0] = n + 1

@njit(cache=True, fastmath=True, error_model='numpy')
def fill_b(a, L, who, idx, off, val, dx, dy, bx, by, bz, vv, tt_, order, tp0):
    for ii in range(L):
        pi = who[a + ii]; si = idx[pi]
        ddx = dx + off[pi, 0]; ddy = dy + off[pi, 1]
        bx[ii] = val[si, 1] + val[si, 11] * ddx + val[si, 12] * ddy
        by[ii] = val[si, 2] + val[si, 13] * ddx + val[si, 14] * ddy
        bz[ii] = val[si, 3] + val[si, 15] * ddx + val[si, 16] * ddy
        vv[ii] = val[si, 0] + val[si, 9] * ddx + val[si, 10] * ddy
        tt_[ii] = val[si, 4] + val[si, 22] * ddx + val[si, 23] * ddy - tp0
    if L <= 48:
        for ii in range(L):
            x = bx[ii]; q = ii - 1
            while q >= 0 and bx[order[q]] > x:
                order[q + 1] = order[q]; q -= 1
            order[q + 1] = ii
    else:
        o = np.argsort(bx[:L])
        for ii in range(L): order[ii] = o[ii]

@njit(cache=True, fastmath=True, error_model='numpy')
def hash_cell(k, cell_start, tid, val, T, narrow, nb_ids, nb_off, cr, tol, tp0, chi_thr, tp_tol, dt_min, out_i, out_j, out_f, cap, rcnt, table, touched, runset, runtouched, slot, slot_touched):
    m = cell_start[k + 1] - cell_start[k]
    for a in range(nb_ids.shape[1]):
        kp = nb_ids[k, a]
        if kp < 0: break
        m += cell_start[kp + 1] - cell_start[kp]
    if m < 2: return 0, 0, 0, 0, 0
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
    if m < 2: return 0, 0, 0, 0, 0
    ext = 1.25 * cr
    tv = 2 * tol[0]; tt = 2 * tol[4]
    # inserts
    K = np.empty(m * NSUB * NSUB, np.int64); W = np.empty(m * NSUB * NSUB, np.int32)
    tot = 0
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
                bv = int(math.floor(v / tv)); bb = int(math.floor(tpv / tt)) + NT // 2
                if bv < 0 or bv >= NV or bb < 0 or bb >= NT: continue
                K[tot] = (j * NV + bv) * NT + bb; W[tot] = p; tot += 1
    if tot < 2: return 0, 0, 0, 0, 0
    K2 = np.empty(tot, np.int64); W2 = np.empty(tot, np.int32)
    radix_sort(K, W, tot, K2, W2, rcnt)
    # run starts
    nrun = 0
    for i in range(tot):
        if i == 0 or K[i] != K[i - 1]: K2[nrun] = i; nrun += 1
    K2[nrun] = tot
    rkey = np.empty(nrun, np.int64)
    for r in range(nrun): rkey[r] = K[K2[r]]
    hmask = (1 << HBITS) - 1; rmask = runset.shape[0] - 1
    st = np.zeros(6, np.int64)   # n, ntouched, nover, nslot, nresets, nemitcalls
    nrt = 0
    x0 = -ext; x1 = ext
    Lmax = 0
    for r in range(nrun):
        L = K2[r + 1] - K2[r]
        if L > Lmax: Lmax = L
    bxA = np.empty(Lmax); byA = np.empty(Lmax); bzA = np.empty(Lmax); vA = np.empty(Lmax); tA = np.empty(Lmax); oA = np.empty(Lmax, np.int64)
    bxB = np.empty(Lmax); byB = np.empty(Lmax); bzB = np.empty(Lmax); vB = np.empty(Lmax); tB = np.empty(Lmax); oB = np.empty(Lmax, np.int64)
    nrunlong = 0
    for r in range(nrun):
        a = K2[r]; b = K2[r + 1]; L = b - a
        key = rkey[r]
        j = key // (NV * NT); bv = (key // NT) % NV; bb = key % NT
        jx = j // NSUB; jy = j % NSUB
        dx = (jx + 0.5) / NSUB * 2 * cr - cr; dy = (jy + 0.5) / NSUB * 2 * cr - cr
        haveA = False
        if L >= 2:
            if L > 256: nrunlong += 1
            # set-hash of this run (order independent)
            hs = np.int64(L)
            for ii in range(L):
                q = np.int64(W[a + ii]); hs += (q + 1) * np.int64(0x9E3779B97F4A7C15)
            hh = (hs ^ (hs >> 29)) & rmask
            skip = False
            while runset[hh] != 0:
                if runset[hh] == hs: skip = True; break
                hh = (hh + 1) & rmask
            if not skip:
                fill_b(a, L, W, idx, off, val, dx, dy, bxA, byA, bzA, vA, tA, oA, tp0); haveA = True
                for oi in range(L):
                    ii = oA[oi]; pi = W[a + ii]; si = idx[pi]; ti = tid[si]
                    for oj in range(oi + 1, L):
                        jj = oA[oj]
                        if bxA[jj] - bxA[ii] > tol[1]: break
                        pj = W[a + jj]; sj = idx[pj]; tj = tid[sj]
                        if tj == ti or abs(T[ti] - T[tj]) < dt_min: continue
                        if abs(byA[ii] - byA[jj]) > tol[2] or abs(bzA[ii] - bzA[jj]) > tol[3]: continue
                        if abs(vA[ii] - vA[jj]) > tol[0] or abs(tA[ii] - tA[jj]) > tol[4]: continue
                        emit(si, sj, ti, tj, val, x0, x1, chi_thr, tp_tol, table, hmask, touched, st, out_i, out_j, out_f, cap, slot, slot_touched)
                # fully degenerate run (all within tol in b): remember so later sub-cells don't re-sweep it
                if L > 2 and bxA[oA[L - 1]] - bxA[oA[0]] <= tol[1]:
                    ymin = byA[0]; ymax = byA[0]; zmin = bzA[0]; zmax = bzA[0]
                    for ii in range(1, L):
                        if byA[ii] < ymin: ymin = byA[ii]
                        if byA[ii] > ymax: ymax = byA[ii]
                        if bzA[ii] < zmin: zmin = bzA[ii]
                        if bzA[ii] > zmax: zmax = bzA[ii]
                    if ymax - ymin <= tol[2] and zmax - zmin <= tol[3] and nrt < runtouched.shape[0] and runset[hh] == 0:
                        runset[hh] = hs; runtouched[nrt] = hh; nrt += 1
        # neighbour buckets: +1 tp, +1 vinf (-1,0,+1 tp)
        for nbk in range(4):
            if nbk == 0:
                if bb + 1 >= NT: continue
                nkey = key + 1
            elif nbk == 1:
                if bv + 1 >= NV or bb - 1 < 0: continue
                nkey = key + NT - 1
            elif nbk == 2:
                if bv + 1 >= NV: continue
                nkey = key + NT
            else:
                if bv + 1 >= NV or bb + 1 >= NT: continue
                nkey = key + NT + 1
            # find run with nkey (sorted rkey; nkey > key so search forward)
            lo = r + 1; hi = nrun
            while lo < hi:
                mid = (lo + hi) >> 1
                if rkey[mid] < nkey: lo = mid + 1
                else: hi = mid
            if lo >= nrun or rkey[lo] != nkey: continue
            a2 = K2[lo]; b2 = K2[lo + 1]; L2 = b2 - a2
            if not haveA:
                fill_b(a, L, W, idx, off, val, dx, dy, bxA, byA, bzA, vA, tA, oA, tp0); haveA = True
            fill_b(a2, L2, W, idx, off, val, dx, dy, bxB, byB, bzB, vB, tB, oB, tp0)
            start = 0
            for oi in range(L):
                ii = oA[oi]; pi = W[a + ii]; si = idx[pi]; ti = tid[si]; xi = bxA[ii]
                while start < L2 and bxB[oB[start]] < xi - tol[1]: start += 1
                for oj in range(start, L2):
                    jj = oB[oj]
                    if bxB[jj] - xi > tol[1]: break
                    pj = W[a2 + jj]; sj = idx[pj]; tj = tid[sj]
                    if tj == ti or abs(T[ti] - T[tj]) < dt_min: continue
                    if abs(byA[ii] - byB[jj]) > tol[2] or abs(bzA[ii] - bzB[jj]) > tol[3]: continue
                    if abs(vA[ii] - vB[jj]) > tol[0] or abs(tA[ii] - tB[jj]) > tol[4]: continue
                    emit(si, sj, ti, tj, val, x0, x1, chi_thr, tp_tol, table, hmask, touched, st, out_i, out_j, out_f, cap, slot, slot_touched)
    # clear tables
    for i in range(st[1]): table[touched[i]] = -1
    for i in range(nrt): runset[runtouched[i]] = 0
    if st[3] < slot_touched.shape[0]:
        for i in range(st[3]): slot[slot_touched[i], 0] = -1
    else:
        slot[:, 0] = -1
    return st[0], st[2], nrunlong, st[1] + st[4] * touched.shape[0], st[5]

@njit(parallel=True, cache=True, error_model='numpy')
def hash_cells(k0, k1, cell_start, tid, val, T, narrow, nb_ids, nb_off, cr, tol, tp0, chi_thr, tp_tol, dt_min, cap, maxrun):
    nk = k1 - k0
    out_i = np.empty((nk, cap), np.int64); out_j = np.empty((nk, cap), np.int64); out_f = np.empty((nk, cap, 4), np.float32)
    cnt = np.zeros(nk, np.int64); over = np.zeros(nk, np.int64); rover = np.zeros(nk, np.int64)
    nthr = 8
    rcnt = np.empty((nthr, RMASK + 2), np.int64)
    table = np.full((nthr, 1 << HBITS), -1, np.int64); touched = np.empty((nthr, 1 << TBITS), np.int64)
    runset = np.zeros((nthr, 1 << 16), np.int64); runtouched = np.empty((nthr, 1 << 15), np.int64)
    slot = np.full((nthr, 1 << HBITS, 2), -1, np.int64); slot_touched = np.empty((nthr, 1 << TBITS), np.int64)
    for th in prange(nthr):
        for a in range(th, nk, nthr):
            n, nov, nro, _, _ = hash_cell(k0 + a, cell_start, tid, val, T, narrow, nb_ids, nb_off, cr, tol, tp0, chi_thr, tp_tol, dt_min, out_i[a], out_j[a], out_f[a], cap, rcnt[th], table[th], touched[th], runset[th], runtouched[th], slot[th], slot_touched[th])
            cnt[a] = n; over[a] = nov; rover[a] = nro
    return cnt, over, rover, out_i, out_j, out_f

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
        print(f'pass {p}: bands {B0}-{B1}, {len(cell)} sols, kernel {t1-t0:.0f}s, pairs {len(P)} ({time.time()-t1:.0f}s) overflow {ov} runs>256 {ro}', flush=True)
        del cell, tid, val
    return np.concatenate(allP), over, nsol

# ---------------- compiled triangle clustering (same semantics as search4.triangle_clusters) ----------------
@njit(cache=True, error_model='numpy')
def _tri_cell(a, b, d0, d1, n, thr):
    """edges (a,b) with node ids 0..n-1; keep edge if it lies in a triangle whose three (d0,d1) agree within thr (max abs dev from mean)."""
    m = len(a)
    deg = np.zeros(n + 1, np.int64)
    for r in range(m): deg[a[r] + 1] += 1; deg[b[r] + 1] += 1
    for i in range(n): deg[i + 1] += deg[i]
    nbr = np.empty(2 * m, np.int64); eid = np.empty(2 * m, np.int64); pos = deg[:n].copy()
    for r in range(m):
        nbr[pos[a[r]]] = b[r]; eid[pos[a[r]]] = r; pos[a[r]] += 1
        nbr[pos[b[r]]] = a[r]; eid[pos[b[r]]] = r; pos[b[r]] += 1
    for i in range(n):   # sort each adjacency list by neighbour id (insertion sort for short lists, argsort for long: dense cells have deg ~1e3-1e4)
        s = deg[i]; e = deg[i + 1]
        if e - s <= 32:
            for p in range(s + 1, e):
                v = nbr[p]; w = eid[p]; q = p - 1
                while q >= s and nbr[q] > v:
                    nbr[q + 1] = nbr[q]; eid[q + 1] = eid[q]; q -= 1
                nbr[q + 1] = v; eid[q + 1] = w
        else:
            o = np.argsort(nbr[s:e]); t1 = nbr[s:e][o]; t2 = eid[s:e][o]
            for p in range(e - s): nbr[s + p] = t1[p]; eid[s + p] = t2[p]
    keep = np.zeros(m, np.uint8)
    for r in range(m):
        x = a[r]; y = b[r]
        if deg[x + 1] - deg[x] > deg[y + 1] - deg[y]: x, y = y, x
        ys = deg[y]; ye = deg[y + 1]
        for p in range(deg[x], deg[x + 1]):
            c = nbr[p]
            if c == y: continue
            lo = ys; hi = ye
            while lo < hi:
                mid = (lo + hi) >> 1
                if nbr[mid] < c: lo = mid + 1
                else: hi = mid
            if lo >= ye or nbr[lo] != c: continue
            r2 = eid[p]; r3 = eid[lo]
            m0 = (d0[r] + d0[r2] + d0[r3]) / 3.0; m1 = (d1[r] + d1[r2] + d1[r3]) / 3.0
            dev = max(abs(d0[r] - m0), abs(d0[r2] - m0), abs(d0[r3] - m0), abs(d1[r] - m1), abs(d1[r2] - m1), abs(d1[r3] - m1))
            if dev < thr: keep[r] = 1; keep[r2] = 1; keep[r3] = 1
    # union-find over kept edges
    parent = np.arange(n)
    for r in range(m):
        if keep[r] == 0: continue
        x = a[r]
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        y = b[r]
        while parent[y] != y: parent[y] = parent[parent[y]]; y = parent[y]
        if x != y: parent[x] = y
    lab = np.empty(n, np.int64)
    for i in range(n):
        x = i
        while parent[x] != x: x = parent[x]
        lab[i] = x
    return keep, lab

@njit(cache=True, error_model='numpy')
def _components(a, b, n):
    parent = np.arange(n)
    for r in range(len(a)):
        x = a[r]
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        y = b[r]
        while parent[y] != y: parent[y] = parent[parent[y]]; y = parent[y]
        if x != y: parent[x] = y
    lab = np.empty(n, np.int64)
    for i in range(n):
        x = i
        while parent[x] != x: x = parent[x]
        lab[i] = x
    return lab

def triangle_clusters(P, cr, dmax=0.35, dense=50000):
    out = []
    o = np.argsort(P[:, 2], kind='stable'); P = P[o]; cells = P[:, 2].astype(np.int64)
    bounds = np.r_[0, np.flatnonzero(np.diff(cells)) + 1, len(P)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 3: continue
        Q = P[s:e]
        nodes, inv = np.unique(np.r_[Q[:, 0], Q[:, 1]].astype(np.int64), return_inverse=True)
        a = inv[:len(Q)].astype(np.int64); b = inv[len(Q):].astype(np.int64)
        if e - s > dense:   # too many edges for triangle enumeration (cubic in near-cliques): components only, flagged by cell
            keep = np.ones(len(Q), np.uint8); lab = _components(a, b, len(nodes))
        else:
            keep, lab = _tri_cell(a, b, np.ascontiguousarray(Q[:, 5]), np.ascontiguousarray(Q[:, 6]), len(nodes), dmax * cr)
        if not keep.any(): continue
        touched = np.zeros(len(nodes), bool); touched[a[keep == 1]] = True; touched[b[keep == 1]] = True
        Qk = Q[keep == 1]
        for rt in np.unique(lab[touched]):
            mem = nodes[(lab == rt) & touched]
            if len(mem) >= 3: out.append((int(cells[s]), mem, float(Qk[:, 5].mean()), float(Qk[:, 6].mean())))
    return out
