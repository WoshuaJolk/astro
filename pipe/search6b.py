"""Stage 2 v4b (search6 + per-thread preallocated insert/sort buffers, int32 keys): search5's pair criterion (within TOL in all 5 dims at some common sub-cell sample), made cache-local:
 - per cell, the ~0.5M candidate solutions (cell + neighbours) are gathered once into a contiguous float32 block V
   (val is a 7.7 GB float64 array; random row reads from it dominated search5's time)
 - no solution-pair dedupe table (32 MB random probes); a redundant exact test on local data is cheaper.
   Output semantics unchanged: slot map keeps the min-chi solution row per tracklet pair
 - slot table sized to the output cap (2^19 entries, 6 MB) instead of 64 MB"""
import numpy as np, math, time
from numba import njit, prange
from search3 import run_kernel, neighbours, dedupe, components
from search4 import narrow_flags, TOL, NSUB, NV, NT
from search5 import radix_sort, RMASK, triangle_clusters
SBITS = 19

@njit(cache=True, fastmath=True, error_model='numpy')
def exact_pair_l(sa, sb, V, x0, x1, chi_thr):
    h00 = 0.0; h01 = 0.0; h11 = 0.0; g0 = 0.0; g1 = 0.0
    for c in range(5):
        oi = 9 + 2 * c if c < 4 else 22
        w = 1.0 / (V[sa, 17 + c] * V[sa, 17 + c] + V[sb, 17 + c] * V[sb, 17 + c] + 1e-12)
        de = V[sa, c] - V[sb, c]
        a0 = V[sa, oi] - V[sb, oi]; a1 = V[sa, oi + 1] - V[sb, oi + 1]
        h00 += w * a0 * a0; h01 += w * a0 * a1; h11 += w * a1 * a1; g0 += w * a0 * de; g1 += w * a1 * de
    det = h00 * h11 - h01 * h01
    if det > 1e-20:
        d0 = -(h11 * g0 - h01 * g1) / det; d1 = -(h00 * g1 - h01 * g0) / det
    else:
        d0 = 0.5 * (x0 + x1); d1 = d0
    if d0 < x0: d0 = x0
    if d0 > x1: d0 = x1
    if d1 < x0: d1 = x0
    if d1 > x1: d1 = x1
    chi = 0.0; dtp = 0.0
    for c in range(5):
        oi = 9 + 2 * c if c < 4 else 22
        w = 1.0 / (V[sa, 17 + c] * V[sa, 17 + c] + V[sb, 17 + c] * V[sb, 17 + c] + 1e-12)
        r = (V[sa, c] - V[sb, c]) + (V[sa, oi] - V[sb, oi]) * d0 + (V[sa, oi + 1] - V[sb, oi + 1]) * d1
        chi += w * r * r
        if c == 4: dtp = abs(r)
    if chi > chi_thr: return -1.0, d0, d1, dtp
    return chi, d0, d1, dtp

@njit(cache=True, fastmath=True, error_model='numpy')
def emit_l(pi, pj, ti, tj, V, idx, x0, x1, chi_thr, out_i, out_j, out_f, cap, skey, sidx, smask, stouched, st):
    st[5] += 1
    chi, d0, d1, dtp = exact_pair_l(pi, pj, V, x0, x1, chi_thr)
    if chi < 0: return
    lo_ = ti if ti < tj else tj; hi_ = tj if ti < tj else ti
    tkey = lo_ * 1000003 + hi_
    g = ((tkey ^ (tkey >> 29)) * 11400714819323198485) & smask
    while skey[g] != -1:
        if skey[g] == tkey:
            n = sidx[g]
            if chi < out_f[n, 0]:
                out_i[n] = idx[pi]; out_j[n] = idx[pj]; out_f[n, 0] = chi; out_f[n, 1] = d0; out_f[n, 2] = d1; out_f[n, 3] = dtp
            return
        g = (g + 1) & smask
    n = st[0]
    if n >= cap: st[2] += 1; return
    skey[g] = tkey; sidx[g] = n; stouched[n] = g
    out_i[n] = idx[pi]; out_j[n] = idx[pj]; out_f[n, 0] = chi; out_f[n, 1] = d0; out_f[n, 2] = d1; out_f[n, 3] = dtp; st[0] = n + 1

@njit(cache=True, fastmath=True, error_model='numpy')
def fill_l(a, L, who, off, V, dx, dy, bx, by, bz, vv, tt_, order, tp0):
    for ii in range(L):
        p = who[a + ii]
        ddx = dx + off[p, 0]; ddy = dy + off[p, 1]
        bx[ii] = V[p, 1] + V[p, 11] * ddx + V[p, 12] * ddy
        by[ii] = V[p, 2] + V[p, 13] * ddx + V[p, 14] * ddy
        bz[ii] = V[p, 3] + V[p, 15] * ddx + V[p, 16] * ddy
        vv[ii] = V[p, 0] + V[p, 9] * ddx + V[p, 10] * ddy
        tt_[ii] = V[p, 4] + V[p, 22] * ddx + V[p, 23] * ddy - tp0
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
def hash_cell(k, cell_start, tid, val, T, narrow, nb_ids, nb_off, cr, tol, tp0, chi_thr, tp_tol, dt_min, out_i, out_j, out_f, cap, rcnt, runset, runtouched, skey, sidx, stouched, Kb, Wb, K2b, W2b):
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
    V = np.empty((m, 24), np.float32)
    for p in range(m):
        s = idx[p]
        for c in range(24): V[p, c] = val[s, c]
    tl = np.empty(m, np.int64); TT = np.empty(m)
    for p in range(m): tl[p] = tid[idx[p]]; TT[p] = T[tl[p]]
    ext = 1.25 * cr
    tv = 2 * tol[0]; tt = 2 * tol[4]
    if Kb.shape[0] >= m * NSUB * NSUB:
        K = Kb; W = Wb
    else:
        K = np.empty(m * NSUB * NSUB, np.int32); W = np.empty(m * NSUB * NSUB, np.int32)
    tot = 0
    for p in range(m):
        for jx in range(NSUB):
            dx = (jx + 0.5) / NSUB * 2 * cr - cr
            ddx = dx + off[p, 0]
            if abs(ddx) > ext: continue
            for jy in range(NSUB):
                dy = (jy + 0.5) / NSUB * 2 * cr - cr
                ddy = dy + off[p, 1]
                if abs(ddy) > ext: continue
                j = jx * NSUB + jy
                v = V[p, 0] + V[p, 9] * ddx + V[p, 10] * ddy
                tpv = V[p, 4] + V[p, 22] * ddx + V[p, 23] * ddy - tp0
                bv = int(math.floor(v / tv)); bb = int(math.floor(tpv / tt)) + NT // 2
                if bv < 0 or bv >= NV or bb < 0 or bb >= NT: continue
                K[tot] = (j * NV + bv) * NT + bb; W[tot] = p; tot += 1
    if tot < 2: return 0, 0, 0, 0, 0
    if K2b.shape[0] > tot:
        K2 = K2b; W2 = W2b
    else:
        K2 = np.empty(tot + 1, np.int32); W2 = np.empty(tot + 1, np.int32)
    radix_sort(K, W, tot, K2, W2, rcnt)
    nrun = 0
    for i in range(tot):
        if i == 0 or K[i] != K[i - 1]: K2[nrun] = i; nrun += 1
    K2[nrun] = tot
    rkey = np.empty(nrun, np.int64)
    for r in range(nrun): rkey[r] = np.int64(K[K2[r]])
    smask = skey.shape[0] - 1; rmask = runset.shape[0] - 1
    st = np.zeros(6, np.int64)   # n, -, nover, -, -, nemitcalls
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
            hs = np.int64(L)
            for ii in range(L):
                q = np.int64(W[a + ii]); hs += (q + 1) * np.int64(0x9E3779B97F4A7C15)
            hh = (hs ^ (hs >> 29)) & rmask
            skip = False
            while runset[hh] != 0:
                if runset[hh] == hs: skip = True; break
                hh = (hh + 1) & rmask
            if not skip:
                fill_l(a, L, W, off, V, dx, dy, bxA, byA, bzA, vA, tA, oA, tp0); haveA = True
                for oi in range(L):
                    ii = oA[oi]; pi = W[a + ii]; ti = tl[pi]; Ti = TT[pi]
                    for oj in range(oi + 1, L):
                        jj = oA[oj]
                        if bxA[jj] - bxA[ii] > tol[1]: break
                        pj = W[a + jj]; tj = tl[pj]
                        if tj == ti or abs(Ti - TT[pj]) < dt_min: continue
                        if abs(byA[ii] - byA[jj]) > tol[2] or abs(bzA[ii] - bzA[jj]) > tol[3]: continue
                        if abs(vA[ii] - vA[jj]) > tol[0] or abs(tA[ii] - tA[jj]) > tol[4]: continue
                        emit_l(pi, pj, ti, tj, V, idx, x0, x1, chi_thr, out_i, out_j, out_f, cap, skey, sidx, smask, stouched, st)
                if L > 2 and bxA[oA[L - 1]] - bxA[oA[0]] <= tol[1]:
                    ymin = byA[0]; ymax = byA[0]; zmin = bzA[0]; zmax = bzA[0]
                    for ii in range(1, L):
                        if byA[ii] < ymin: ymin = byA[ii]
                        if byA[ii] > ymax: ymax = byA[ii]
                        if bzA[ii] < zmin: zmin = bzA[ii]
                        if bzA[ii] > zmax: zmax = bzA[ii]
                    if ymax - ymin <= tol[2] and zmax - zmin <= tol[3] and nrt < runtouched.shape[0] and runset[hh] == 0:
                        runset[hh] = hs; runtouched[nrt] = hh; nrt += 1
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
            lo = r + 1; hi = nrun
            while lo < hi:
                mid = (lo + hi) >> 1
                if rkey[mid] < nkey: lo = mid + 1
                else: hi = mid
            if lo >= nrun or rkey[lo] != nkey: continue
            a2 = K2[lo]; b2 = K2[lo + 1]; L2 = b2 - a2
            if not haveA:
                fill_l(a, L, W, off, V, dx, dy, bxA, byA, bzA, vA, tA, oA, tp0); haveA = True
            fill_l(a2, L2, W, off, V, dx, dy, bxB, byB, bzB, vB, tB, oB, tp0)
            start = 0
            for oi in range(L):
                ii = oA[oi]; pi = W[a + ii]; ti = tl[pi]; Ti = TT[pi]; xi = bxA[ii]
                while start < L2 and bxB[oB[start]] < xi - tol[1]: start += 1
                for oj in range(start, L2):
                    jj = oB[oj]
                    if bxB[jj] - xi > tol[1]: break
                    pj = W[a2 + jj]; tj = tl[pj]
                    if tj == ti or abs(Ti - TT[pj]) < dt_min: continue
                    if abs(byA[ii] - byB[jj]) > tol[2] or abs(bzA[ii] - bzB[jj]) > tol[3]: continue
                    if abs(vA[ii] - vB[jj]) > tol[0] or abs(tA[ii] - tB[jj]) > tol[4]: continue
                    emit_l(pi, pj, ti, tj, V, idx, x0, x1, chi_thr, out_i, out_j, out_f, cap, skey, sidx, smask, stouched, st)
    for i in range(nrt): runset[runtouched[i]] = 0
    for i in range(st[0]): skey[stouched[i]] = -1
    return st[0], st[2], nrunlong, 0, st[5]

@njit(parallel=True, cache=True, error_model='numpy')
def hash_cells(klist, cell_start, tid, val, T, narrow, nb_ids, nb_off, cr, tol, tp0, chi_thr, tp_tol, dt_min, cap, maxrun):
    nk = klist.shape[0]
    out_i = np.empty((nk, cap), np.int64); out_j = np.empty((nk, cap), np.int64); out_f = np.empty((nk, cap, 4), np.float32)
    cnt = np.zeros(nk, np.int64); over = np.zeros(nk, np.int64); rover = np.zeros(nk, np.int64)
    nthr = 8
    rcnt = np.empty((nthr, RMASK + 2), np.int64)
    runset = np.zeros((nthr, 1 << 16), np.int64); runtouched = np.empty((nthr, 1 << 15), np.int64)
    sb = 1 << SBITS
    while sb < 2 * cap: sb *= 2
    skey = np.full((nthr, sb), -1, np.int64); sidx = np.empty((nthr, sb), np.int32); stouched = np.empty((nthr, cap), np.int64)
    ncell = cell_start.shape[0] - 1
    ncnt = np.zeros(ncell, np.int64)
    for k in range(ncell):
        c = 0
        for s_ in range(cell_start[k], cell_start[k + 1]): c += narrow[s_]
        ncnt[k] = c
    mmax = 0
    for a_ in range(nk):
        k = klist[a_]
        m = ncnt[k]
        for a in range(nb_ids.shape[1]):
            kp = nb_ids[k, a]
            if kp < 0: break
            m += ncnt[kp]
        if m > mmax: mmax = m
    nbuf = mmax * NSUB * NSUB + 1
    Kb = np.empty((nthr, nbuf), np.int32); Wb = np.empty((nthr, nbuf), np.int32); K2b = np.empty((nthr, nbuf), np.int32); W2b = np.empty((nthr, nbuf), np.int32)
    for th in prange(nthr):
        for a in range(th, nk, nthr):
            n, nov, nro, _, _ = hash_cell(klist[a], cell_start, tid, val, T, narrow, nb_ids, nb_off, cr, tol, tp0, chi_thr, tp_tol, dt_min, out_i[a], out_j[a], out_f[a], cap, rcnt[th], runset[th], runtouched[th], skey[th], sidx[th], stouched[th], Kb[th], Wb[th], K2b[th], W2b[th])
            cnt[a] = n; over[a] = nov; rover[a] = nro
    return cnt, over, rover, out_i, out_j, out_f

def find_pairs(cell, tid, val, T, g, nb_ids, nb_off, kfrom=0, kto=None, tp0=0.0, chi_thr=20.0, tp_tol=10.0, dt_min=1.0, cap=200000, maxrun=256, kchunk=256, retry_cap=4000000, retry_batch=8):
    nretry = 0
    cell_start = np.searchsorted(cell, np.arange(g['ncell'] + 1)).astype(np.int64)
    narrow = narrow_flags(val)
    if kto is None: kto = g['ncell']
    P = []; tot_over = 0; tot_rover = 0
    for k0 in range(kfrom, kto, kchunk):
        k1 = min(kto, k0 + kchunk)
        cnt, over, rover, oi, oj, of = hash_cells(np.arange(k0, k1, dtype=np.int64), cell_start, tid, val, T, narrow, nb_ids, nb_off, g['cell_rad'], TOL, tp0, chi_thr, tp_tol, dt_min, cap, maxrun)
        tot_rover += rover.sum()
        bad = np.flatnonzero(over > 0) if retry_cap > cap else np.zeros(0, np.int64)
        redo = {}
        for b0 in range(0, len(bad), retry_batch):  # lossless retry of overflowed cells, 8 at a time in parallel, with a much larger buffer
            bb = bad[b0:b0 + retry_batch]
            cnt2, over2, rover2, oi2, oj2, of2 = hash_cells((k0 + bb).astype(np.int64), cell_start, tid, val, T, narrow, nb_ids, nb_off, g['cell_rad'], TOL, tp0, chi_thr, tp_tol, dt_min, retry_cap, maxrun)
            for q, a in enumerate(bb): redo[a] = (cnt2[q], oi2[q], oj2[q], of2[q], over2[q])
            nretry += len(bb)
        for a in range(k1 - k0):
            if a in redo: n, ci, cj, cf, ov_ = redo[a]
            else: n, ci, cj, cf, ov_ = cnt[a], oi[a], oj[a], of[a], over[a]
            tot_over += ov_
            if n: P.append(np.column_stack([tid[ci[:n]], tid[cj[:n]], np.full(n, k0 + a), cf[:n, 0], cf[:n, 3], cf[:n, 1], cf[:n, 2]]))
    P = np.concatenate(P) if P else np.zeros((0, 7))
    if nretry: print(f'    retried {nretry} overflowed cells with cap {retry_cap}; residual overflow {tot_over}', flush=True)
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
