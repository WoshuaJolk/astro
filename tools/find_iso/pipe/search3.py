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

_nb_cache = {}
def _kernel_chunks(tr, g, cb, blo=0, bhi=1 << 30, nl=40, nr=56, step=1.5, vmin=8.0, vmax=80.0, qmax=6.0, maxc=6000, chunk=2048, keep_cells=None, seed_all=False):
    """Yields (cell, tid, val) per tracklet chunk for solutions whose radiant cell is in sky bands [blo, bhi) (and in keep_cells if given)."""
    lr, rd = K4.default_scan(nl=nl, nr=nr)
    n = len(tr['t']); bhi = min(bhi, g['nb'])
    k0 = g['starts'][blo]; k1 = g['starts'][bhi] if bhi < g['nb'] else g['ncell']
    if keep_cells is None or seed_all: keep_u8 = np.ones(g['ncell'], np.uint8)   # seed_all: refine every cell in the bands, filter output only
    else:   # the kernel seeds a solution in cell kc but may assign it to the neighbouring cell its refined radiant falls in, so seed
        # every kept cell AND its neighbour ring (dilated mask); the output is still filtered to keep_cells below
        nbi = _nb_cache.get(id(g))
        if nbi is None: nbi = neighbours(g)[0]; _nb_cache[id(g)] = nbi
        keep_u8 = keep_cells.astype(np.uint8).copy(); nn = nbi[keep_cells].ravel(); keep_u8[nn[nn >= 0]] = 1
    for s in range(0, n, chunk):
        ii = np.arange(s, min(n, s + chunk))
        c, rc, rv, st = K4.process_chunk(tr['t'][ii], tr['u'][ii], tr['udot'][ii], tr['obs_pos'][ii], tr['obs_vel'][ii], tr['sig_rate'][ii],
                                         lr, rd, g['nb'], g['nras'], g['starts'], g['centers'], cb, g['cell_rad'], vmin, vmax, qmax, step, maxc, 0, blo, bhi, keep_u8)
        tot = int(c.sum()); cc = np.empty(tot, np.int32); tt = np.empty(tot, np.int32); vv = np.empty((tot, NOUT), np.float32); p = 0
        for a in range(len(ii)):
            k = c[a]
            cc[p:p + k] = rc[a, :k]; tt[p:p + k] = ii[a]; vv[p:p + k] = rv[a, :k * NOUT].reshape(k, NOUT); p += k
        keep = keep_cells[cc] if keep_cells is not None else (cc >= k0) & (cc < k1)   # keep_cells: bool mask (ncell,) of cells a pass actually reads
        yield cc[keep], tt[keep], vv[keep]


def run_kernel(tr, g, cb, blo=0, bhi=1 << 30, keep_cells=None, **kw):
    """kernel over all tracklets, keeping only solutions whose radiant cell lies in sky bands [blo, bhi). Sorted by cell."""
    cells = []; tids = []; vals = []
    for cc, tt, vv in _kernel_chunks(tr, g, cb, blo, bhi, keep_cells=keep_cells, **kw):
        cells.append(cc); tids.append(tt); vals.append(vv)
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


def build_bands(tr, g, cb, bands, cache_dir, tag, bpad=2, **kw):
    """One kernel sweep over all tracklets for several sky bands at once (seeds from bpad bands beyond the group, output filtered to the
    group), streamed to raw per-band files so peak memory is ~one band; then each band is cell-sorted and saved as .npy."""
    import os
    bands = sorted(int(b) for b in bands); nb = g['nb']
    km = np.zeros(g['ncell'], bool)
    for b in bands: km[g['starts'][b]:(g['starts'][b + 1] if b + 1 < nb else g['ncell'])] = True
    band_of = np.searchsorted(g['starts'], np.arange(g['ncell']), 'right') - 1
    fh = {b: [open(f'{cache_dir}/{tag}_b{b:03d}_{s}.raw', 'wb') for s in ('cell', 'tid', 'val')] for b in bands}
    # a wider sweep finds more solutions per tracklet than a per-pass sweep, so the per-tracklet cap (maxc, default 6000) must grow with it
    # or the highest bands of the sweep get truncated; chunk shrinks to keep the kernel's scratch (chunk x maxc x NOUT floats) constant
    nsweep = min(bands[-1] + bpad + 1, nb) - max(bands[0] - bpad, 0)
    kw.setdefault('maxc', 6000 * max(1, int(np.ceil(nsweep / 4.0)))); kw.setdefault('chunk', max(256, 2048 * 6000 // kw['maxc']))
    for cc, tt, vv in _kernel_chunks(tr, g, cb, max(bands[0] - bpad, 0), min(bands[-1] + bpad + 1, nb), keep_cells=km, seed_all=True, **kw):
        bb = band_of[cc]
        for b in bands:
            m = bb == b
            if m.any(): fh[b][0].write(cc[m].tobytes()); fh[b][1].write(tt[m].tobytes()); fh[b][2].write(vv[m].tobytes())
    for b in bands:
        for f in fh[b]: f.close()
        base = f'{cache_dir}/{tag}_b{b:03d}'
        c = np.fromfile(base + '_cell.raw', np.int32); t_ = np.fromfile(base + '_tid.raw', np.int32); v = np.fromfile(base + '_val.raw', np.float32).reshape(-1, NOUT)
        o = np.argsort(c, kind='stable'); c = c[o]; t_ = t_[o]; _permute_rows_inplace(v, o.astype(np.int64)); del o
        np.save(base + '_cell.npy', c); np.save(base + '_tid.npy', t_); np.save(base + '_val.tmp.npy', v); os.replace(base + '_val.tmp.npy', base + '_val.npy')
        del c, t_, v
        for s in ('cell', 'tid', 'val'): os.remove(f'{base}_{s}.raw')


def run_kernel_cached(tr, g, cb, blo, bhi, keep_cells, cache_dir, tag, nbands=2, bpad=2, **kw):
    """Same solutions as run_kernel(..., keep_cells=keep_cells) for every kept cell (plus the ones the per-pass path only fills partially),
    but the kernel runs once per group of nbands sky bands and is cached on disk (cell-sorted .npy per band), so consecutive passes that
    share bands do not recompute it. Band files below the lowest kept band are deleted (passes advance monotonically)."""
    import os
    os.makedirs(cache_dir, exist_ok=True)
    nb = g['nb']
    blo_arg, bhi_arg = int(blo), int(min(bhi, nb))
    kk_all = np.flatnonzero(keep_cells); bands = np.unique(np.searchsorted(g['starts'], kk_all, 'right') - 1)   # every band holding a kept cell
    bands = bands[(bands >= blo_arg - 1) & (bands < bhi_arg + 1)]; blo = int(bands.min())   # per-pass span +1 band each side: covers every cell the per-pass kernel populated
    for fn in os.listdir(cache_dir):
        if fn.startswith(f'{tag}_b') and fn.endswith('_cell.npy'):
            b = int(fn[len(tag) + 2:-9])
            if b < blo:
                for suf in ('cell', 'tid', 'val'):
                    try: os.remove(f'{cache_dir}/{tag}_b{b:03d}_{suf}.npy')
                    except FileNotFoundError: pass
    missing = [int(b) for b in bands if not os.path.exists(f'{cache_dir}/{tag}_b{int(b):03d}_val.npy')]
    if missing:   # build the missing bands (and, when a group has room, the following bands too, since the next passes will need them)
        b0 = missing[0]; grp = list(range(b0, min(b0 + nbands, nb)))
        grp = [b for b in grp if not os.path.exists(f'{cache_dir}/{tag}_b{b:03d}_val.npy')]
        build_bands(tr, g, cb, grp, cache_dir, tag, bpad=bpad, **kw)
        for b in missing:
            if not os.path.exists(f'{cache_dir}/{tag}_b{b:03d}_val.npy'): build_bands(tr, g, cb, [b], cache_dir, tag, bpad=bpad, **kw)
    cells = []; tids = []; vals = []
    for b in bands:
        b = int(b)
        base = f'{cache_dir}/{tag}_b{b:03d}'
        c = np.load(base + '_cell.npy', mmap_mode='r'); t_ = np.load(base + '_tid.npy', mmap_mode='r'); v = np.load(base + '_val.npy', mmap_mode='r')
        k0 = g['starts'][b]; k1 = g['starts'][b + 1] if b + 1 < nb else g['ncell']
        kk = np.flatnonzero(keep_cells[k0:k1]) + k0
        if len(kk) == 0: continue
        cs = np.searchsorted(c, np.r_[kk, kk + 1])   # contiguous runs of a sorted cell array
        lo = cs[:len(kk)]; hi = cs[len(kk):]
        idx = np.concatenate([np.arange(a, z) for a, z in zip(lo, hi) if z > a]) if np.any(hi > lo) else np.zeros(0, np.int64)
        if len(idx) == 0: continue
        cells.append(np.asarray(c[idx])); tids.append(np.asarray(t_[idx])); vals.append(np.asarray(v[idx]))
        del c, t_, v
    if not cells: return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, NOUT), np.float32)
    return np.concatenate(cells), np.concatenate(tids), np.concatenate(vals)
