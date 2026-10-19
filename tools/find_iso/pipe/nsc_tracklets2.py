"""Build tracklets from fetched NSC DR2 DECam measurements (ndet<=2 sources = never matched to a persistent object).
Per same-night revisit group: pair detections between exposures (sep <= 12 deg/day * dt, motion > 3 sigma astrometric so a
stationary repeat is not called a tracklet), chain to >=3 detections when a third exposure exists (linear-motion residual
< 0.7"), require >=3 detections (standard MOPS tracklet rule; a bare pair at DECam depth is indistinguishable from noise). Output: same npz schema as itf_tracklets.npz plus an obs table.
No physics cut: all rates up to the kernel's 12 deg/day, all magnitudes, all sky."""
import numpy as np, pandas as pd, glob, gzip, os, sys, time
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/ubuntu/iso/pipe')
from ingest import parse_obscodes, Earth, gmst_rad, delta_t_days, ERAD_AU
D = '/home/ubuntu/iso/data/nsc'; OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/iso/data/nsc_tracklets.npz'
RATE_MAX = 12.0  # deg/day (kernel limit)
codes = parse_obscodes('/home/ubuntu/iso/data/ObsCodes.html'); earth = Earth('/home/ubuntu/iso/data/de440s.bsp')
lon, rc, rs = codes['W84']
ex = pd.read_csv(f'{D}/revisit_exposures.csv')
have = {}
for f in glob.glob(f'{D}/meas/batch_*.done'):
    for e in open(f).read().split(): have[e] = f[:-5] + '.csv.gz'
ex = ex[ex.exposure.isin(have)]
g = ex.groupby(['night', 'ib', 'jb'])
allex = pd.read_csv(f'{D}/revisit_exposures.csv').groupby(['night', 'ib', 'jb']).size()
groups = [k for k, sub in g if len(sub) == allex[k] and len(sub) >= 3]
print(f'{len(groups)} fully fetched groups', flush=True)
cache = {}
def load_batch(f):
    if f not in cache:
        if len(cache) > 40: cache.clear()
        cache[f] = pd.read_csv(f, compression='gzip')
    return cache[f]
# NSC raerr/decerr are already in arcsec (median 0.15"); do not rescale.
def unit(ra, dec):
    ra = np.deg2rad(ra); dec = np.deg2rad(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)

TRK = f'{D}/trk'; os.makedirs(TRK, exist_ok=True)
CH = 500  # first-exposure detections per pairing chunk (bounds memory in dense fields at ~1 h cadence)
def do_group(gi):
    k = groups[gi]; fo = f'{TRK}/g{int(k[0])}_{int(k[1])}_{int(k[2])}.npz'
    if os.path.exists(fo): return gi, 0, 0, 0.0
    t0 = time.time()
    sub = g.get_group(k).sort_values('mjd')
    frames = []
    for e in sub.exposure:
        b = pd.read_csv(have[e], compression='gzip'); m = b[b.exposure == e]
        if len(m): frames.append(m)
    if len(frames) < 3:
        np.savez(fo, obs=np.zeros((0, 5)), rows=np.zeros((0, 3), int)); return gi, 0, 0, 0.0
    E = [(f.mjd.values[0], unit(f.ra.values, f.dec.values), f) for f in frames]
    E.sort(key=lambda x: x[0])
    obs = []; offs = [0]
    for f in E:
        obs.append(np.stack([f[2].mjd.values, f[2].ra.values, f[2].dec.values, f[2].mag_auto.values, np.hypot(f[2].raerr.values, f[2].decerr.values)], 1))
        offs.append(offs[-1] + len(f[2]))
    offs = np.array(offs)
    PAIR_MIN = 10.0 / 1440
    # need >= 3 epochs mutually >= PAIR_MIN apart, otherwise no confirmable tracklet exists in this group (coverage gap)
    ep = [E[0][0]]
    for e in E:
        if e[0] - ep[-1] >= PAIR_MIN: ep.append(e[0])
    if len(ep) < 3:
        np.savez(fo, obs=np.concatenate(obs), rows=np.zeros((0, len(E)), np.int32)); return gi, 0, 0, 0.0
    trees = [cKDTree(e[1]) for e in E]; err = [o[:, 4] for o in obs]
    trk = []
    # pair exposure i with the first later exposure >= PAIR_MIN apart (closer ones only confirm): a near-simultaneous pair
    # has no usable rate, and its propagated error (sigma * dt_pred/dt_pair) would make the confirmation step accept anything.
    PAIR_MIN = 10.0 / 1440
    for i in range(len(E) - 1):
        js = [j for j in range(i + 1, len(E)) if E[j][0] - E[i][0] >= PAIR_MIN]
        j = js[0] if js else len(E) - 1
        dt = E[j][0] - E[i][0]
        if dt <= 0.5 / 1440 or dt > 0.5: continue
        r = 2 * np.sin(np.deg2rad(RATE_MAX * dt) / 2)
        for c0 in range(0, len(E[i][1]), CH):
            sub_tree = cKDTree(E[i][1][c0:c0 + CH])
            M = sub_tree.sparse_distance_matrix(trees[j], r, output_type='coo_matrix')
            if M.nnz == 0: continue
            ia, jb, sep = M.row + c0, M.col, M.data * 206264.8
            sig = np.hypot(err[i][ia], err[j][jb])
            ok = sep >= 3.0 * sig + 0.3
            ia, jb, sig = ia[ok], jb[ok], sig[ok]
            if len(ia) == 0: continue
            ua = E[i][1][ia]; v = (E[j][1][jb] - ua) / dt
            ex_ = np.full((len(ia), len(E)), -1, np.int32); ex_[:, i] = ia; ex_[:, j] = jb
            # a confirmer must be a distinct epoch (>= PAIR_MIN from both pair members): a detection a minute from member i
            # only re-measures i's position and would let a 2-epoch stationary pair plus any random partner pass as a tracklet
            for kk in list(range(j + 1, len(E))) + list(range(i - 1, -1, -1)):
                dtk = E[kk][0] - E[i][0]
                if abs(dtk) > 0.5 or min(abs(dtk), abs(E[kk][0] - E[j][0])) < PAIR_MIN: continue
                pred = ua + v * dtk; pred /= np.linalg.norm(pred, axis=1)[:, None]
                dd, ii = trees[kk].query(pred, k=1)
                tol = 0.7 + 3.0 * sig * max(abs(dtk) / dt, 1.0)
                hit = dd * 206264.8 < tol
                ex_[hit, kk] = ii[hit]
            nm = (ex_ >= 0).sum(1)
            ex_ = ex_[nm >= 3]
            if len(ex_) == 0: continue
            ex_ = np.where(ex_ >= 0, ex_ + offs[:len(E)][None, :], -1)
            trk.append(np.sort(ex_, axis=1))
    obs = np.concatenate(obs)
    W_ = len(E)
    rows = np.concatenate(trk) if trk else np.zeros((0, W_), np.int32)
    rows = np.unique(rows, axis=0)
    np.savez(fo, obs=obs, rows=rows)
    return gi, len(obs), len(rows), time.time() - t0
if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time(); ndone = 0
    with Pool(int(os.environ.get('NSC_NPROC', '4'))) as pool:
        for gi, no, nt, dt in pool.imap_unordered(do_group, range(len(groups))):
            ndone += 1
            if dt > 60 or ndone % 25 == 0: print(f'group {gi} ({ndone}/{len(groups)}): {no} obs, {nt} tracklets, {dt:.0f}s [{time.time()-t0:.0f}s]', flush=True)
    # assemble
    obs = []; trk = []; base = 0
    for gi in range(len(groups)):
        k = groups[gi]; z = np.load(f'{TRK}/g{int(k[0])}_{int(k[1])}_{int(k[2])}.npz'); o, r = z['obs'], z['rows']
        if len(r):
            obs.append(o); trk.append(np.where(r >= 0, r + base, -1).astype(np.int64)); base += len(o)
    print(f'assembled {len(groups)} groups: {sum(len(r) for r in trk)} tracklets, {time.time()-t0:.0f}s', flush=True)
    obs = np.concatenate(obs) if obs else np.zeros((0, 5))
    # dedupe: identical member sets; then drop tracklets whose member set is a subset of a longer one (same first two members check)
    W_ = max((r.shape[1] for r in trk), default=3)
    rows = np.concatenate([np.pad(r, ((0, 0), (W_ - r.shape[1], 0)), constant_values=-1) for r in trk]) if trk else np.zeros((0, W_), int)
    rows = np.unique(rows, axis=0)
    nm = (rows >= 0).sum(1); order = np.argsort(-nm, kind='stable'); rows = rows[order]
    seen = set(); keep = []
    for r in rows:
        m = tuple(int(x) for x in r if x >= 0)
        if any(p in seen for p in zip(m, m[1:])): continue
        seen.update(zip(m, m[1:])); keep.append(m)
    trk = keep
    print(f'{len(obs)} obs, {len(trk)} tracklets after dedupe; size hist {np.bincount([len(t) for t in trk])}', flush=True)
    n = len(trk); mjd = obs[:, 0]; U = unit(obs[:, 1], obs[:, 2])
    mjd_tt = mjd + delta_t_days(mjd); ep, ev = earth.helio(mjd_tt)
    theta = gmst_rad(mjd) + np.deg2rad(lon)
    geo = np.stack([rc * np.cos(theta) * ERAD_AU, rc * np.sin(theta) * ERAD_AU, np.full(len(mjd), rs * ERAD_AU)], 1)
    P = ep + geo; omega = 2 * np.pi * 1.00273781191135448
    V = ev + omega * np.stack([-geo[:, 1], geo[:, 0], np.zeros(len(mjd))], 1)
    t = np.zeros(n); u0 = np.zeros((n, 3)); ud = np.zeros((n, 3)); P0 = np.zeros((n, 3)); V0 = np.zeros((n, 3))
    nobs = np.zeros(n, int); arc = np.zeros(n); rms = np.zeros(n); mag = np.zeros(n); first = np.zeros(n, int)
    flat = []; tid_of_obs = []
    for i, m in enumerate(trk):
        m = list(m); tm = mjd[m]; tc = tm.mean(); dt = tm - tc
        A = np.stack([np.ones(len(m)), dt], 1)
        cu, *_ = np.linalg.lstsq(A, U[m], rcond=None); cp, *_ = np.linalg.lstsq(A, P[m], rcond=None)
        res = U[m] - A @ cu
        t[i] = tc; u0[i] = cu[0] / np.linalg.norm(cu[0]); ud[i] = cu[1] - u0[i] * (cu[1] @ u0[i]); P0[i] = cp[0]; V0[i] = V[m].mean(0)
        nobs[i] = len(m); arc[i] = tm.max() - tm.min(); rms[i] = np.sqrt((res ** 2).sum(1).mean()) * 206264.8; mag[i] = obs[m, 3].mean()
        first[i] = len(flat); flat += m; tid_of_obs += [i] * len(m)
    np.savez(OUT, t=t, u=u0, udot=ud, obs_pos=P0, obs_vel=V0, nobs=nobs, arc_days=arc, rms_arcsec=rms, mag=mag,
             desig=np.array([b'NSC%08d' % i for i in range(n)], dtype='S12'), station=np.full(n, b'W84', dtype='S3'),
             first_line=first, tid_of_obs=np.array(tid_of_obs), obs_order=np.array(flat), sel=np.ones(n, bool),
             obs_table=obs[np.array(flat, int)] if flat else obs)
    rate = np.rad2deg(np.linalg.norm(ud, axis=1))
    print(f'saved {OUT}: {n} tracklets; rate deg/day percentiles {np.percentile(rate,[10,50,90,99]).round(3) if n else None}; rms<2" frac {(rms<2).mean() if n else 0:.2f}; {time.time()-t0:.0f}s')
