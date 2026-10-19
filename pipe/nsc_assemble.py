# vectorized assembly of NSC per-group tracklet checkpoints -> nsc_tracklets.npz (same keys as itf_tracklets.npz)
import numpy as np, glob, os, sys, time
sys.argv = sys.argv[:1]
import nsc_tracklets2 as N
OUT = os.environ.get('NSC_OUT', '/home/ubuntu/iso/data/nsc/nsc_tracklets.npz')
LIM = int(os.environ.get('NSC_LIM', '0'))
t0 = time.time()
files = sorted(glob.glob(f'{N.TRK}/g*.npz'))
if LIM: files = files[:LIM]
obs = []; trk = []; base = 0; W_ = 3
for f in files:
    z = np.load(f); o, r = z['obs'], z['rows']
    if len(r):
        r = np.sort(np.where(r < 0, np.iinfo(np.int32).max, r), 1).astype(np.int64)  # members ascending, pads to the right
        k = int((r < np.iinfo(np.int32).max).sum(1).max()); r = r[:, :k]
        obs.append(o); trk.append(np.where(r < np.iinfo(np.int32).max, r + base, -1)); base += len(o); W_ = max(W_, k)
obs = np.concatenate(obs); rows = np.concatenate([np.pad(r, ((0, 0), (0, W_ - r.shape[1])), constant_values=-1) for r in trk]); del trk
print(f'{len(files)} files: {len(obs)} obs, {len(rows)} tracklets, W={W_}, {time.time()-t0:.0f}s', flush=True)
rows = np.unique(rows, axis=0)
rows = np.sort(np.where(rows < 0, np.iinfo(np.int64).max, rows), 1)  # members ascending, pads (-1) moved to the end as +inf
nm = (rows < np.iinfo(np.int64).max).sum(1)
rows = np.where(rows == np.iinfo(np.int64).max, -1, rows)
order = np.argsort(-nm, kind='stable'); rows = rows[order]; nm = nm[order]
# greedy subset-drop (same rule as nsc_tracklets2): in length order, a tracklet is kept iff none of its consecutive member pairs is already claimed
NO = len(obs); keep = np.ones(len(rows), bool)
for it in range(20):
    a = rows[:, :-1]; b = rows[:, 1:]; valid = (b >= 0) & keep[:, None]
    tid = np.broadcast_to(np.arange(len(rows))[:, None], a.shape)[valid]; key = (a * NO + b)[valid]
    o = np.lexsort((tid, key)); key, tid = key[o], tid[o]
    first = np.r_[True, key[1:] != key[:-1]]; owner = np.maximum.accumulate(np.where(first, np.arange(len(key)), 0))
    bad = np.unique(tid[tid != tid[owner]])
    if len(bad) == 0: break
    keep[bad] = False
    print(f'  dedupe pass {it}: dropped {len(bad)}, kept {keep.sum()}', flush=True)
rows = rows[keep]; nm = nm[keep]; n = len(rows)
print(f'{NO} obs, {n} tracklets after dedupe; size hist {np.bincount(nm)}; {time.time()-t0:.0f}s', flush=True)
mjd = obs[:, 0]; U = N.unit(obs[:, 1], obs[:, 2])
mjd_tt = mjd + N.delta_t_days(mjd); ep, ev = N.earth.helio(mjd_tt)
theta = N.gmst_rad(mjd) + np.deg2rad(N.lon)
geo = np.stack([N.rc * np.cos(theta) * N.ERAD_AU, N.rc * np.sin(theta) * N.ERAD_AU, np.full(len(mjd), N.rs * N.ERAD_AU)], 1)
P = ep + geo; omega = 2 * np.pi * 1.00273781191135448
V = ev + omega * np.stack([-geo[:, 1], geo[:, 0], np.zeros(len(mjd))], 1)
tc = np.zeros(n); u0 = np.zeros((n, 3)); ud = np.zeros((n, 3)); cp0 = np.zeros((n, 3)); V0 = np.zeros((n, 3)); rms = np.zeros(n); arc = np.zeros(n); mag = np.zeros(n)
CHK = 1000000
for c0 in range(0, n, CHK):
    sl = slice(c0, min(n, c0 + CHK)); R = rows[sl]; R = R[:, :int(nm[sl].max())]
    m = R >= 0; idx = np.where(m, R, 0); w = m.astype(float); nmc = nm[sl]
    tm = np.where(m, mjd[idx], 0.0); tcc = tm.sum(1) / nmc; dt = np.where(m, tm - tcc[:, None], 0.0); s2 = (dt ** 2).sum(1)
    def fit(X):
        Xm = X[idx] * w[..., None]; a0 = Xm.sum(1) / nmc[:, None]; a1 = (dt[..., None] * Xm).sum(1) / s2[:, None]
        return a0, a1
    cu0, cu1 = fit(U); cp0[sl], _ = fit(P)
    res = (U[idx] - cu0[:, None, :] - dt[..., None] * cu1[:, None, :]) * w[..., None]
    rms[sl] = np.sqrt((res ** 2).sum((1, 2)) / nmc) * 206264.8
    uu = cu0 / np.linalg.norm(cu0, axis=1, keepdims=True); u0[sl] = uu; ud[sl] = cu1 - uu * (cu1 * uu).sum(1, keepdims=True)
    V0[sl] = (V[idx] * w[..., None]).sum(1) / nmc[:, None]; tc[sl] = tcc
    arc[sl] = np.where(m, tm, -np.inf).max(1) - np.where(m, tm, np.inf).min(1)
    mag[sl] = (np.where(m, obs[idx, 3], 0.0)).sum(1) / nmc
m = rows >= 0
first = np.r_[0, np.cumsum(nm)[:-1]]; flat = rows[m]; tid_of_obs = np.repeat(np.arange(n), nm)
np.savez(OUT, t=tc, u=u0, udot=ud, obs_pos=cp0, obs_vel=V0, nobs=nm, arc_days=arc, rms_arcsec=rms, mag=mag,
         desig=np.array([b'NSC%08d' % i for i in range(n)], dtype='S12'), station=np.full(n, b'W84', dtype='S3'),
         first_line=first, tid_of_obs=tid_of_obs, obs_order=flat, sel=np.ones(n, bool), obs=obs)
print(f'wrote {OUT}: {n} tracklets; rate deg/day q50/q90/max {np.percentile(np.linalg.norm(ud,axis=1)*57.29578,[50,90,100])}; rms" q50/q90 {np.percentile(rms,[50,90])}; {time.time()-t0:.0f}s', flush=True)
