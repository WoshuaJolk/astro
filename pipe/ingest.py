"""Parse MPC 80-col observations -> tracklets with attributables (equatorial J2000, heliocentric observer state).
Output: npz with per-tracklet arrays."""
import numpy as np, sys, re, time
from jplephem.spk import SPK

AU_KM = 149597870.7
ERAD_AU = 6378.137 / AU_KM
DAY_S = 86400.0

def parse_obscodes(path):
    codes = {}
    for line in open(path, encoding='latin-1'):
        if len(line) < 30 or not re.match(r'^[0-9A-Za-z]{3}[ ]', line): continue
        code = line[0:3]
        try:
            lon = float(line[4:13]); rc = float(line[13:21]); rs = float(line[21:30])
        except ValueError:
            if line[4:30].strip() == '':
                codes[code] = (0.0, 0.0, 0.0)
            continue
        codes[code] = (lon, rc, rs)
    return codes

def mjd_from_ymd(y, m, d):
    # d may be fractional
    y = y.astype(np.int64); m = m.astype(np.int64)
    a = (14 - m) // 12
    yy = y + 4800 - a
    mm = m + 12 * a - 3
    jdn = np.floor(d).astype(np.int64) + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    return jdn - 2400001.0 + (d - np.floor(d))  # MJD = JD - 2400000.5, JDN is noon-based: JD = JDN - 0.5 + frac

def delta_t_days(mjd):
    # TT - UTC in days, rough leap-second table (sufficient for linking)
    yr = 1858.879 + mjd / 365.25
    dt = np.full_like(mjd, 69.184)
    dt[yr < 2017.0] = 68.184; dt[yr < 2015.5] = 67.184; dt[yr < 2012.5] = 66.184; dt[yr < 2009.0] = 65.184
    dt[yr < 2006.0] = 64.184; dt[yr < 1999.0] = 63.184; dt[yr < 1997.5] = 62.184; dt[yr < 1996.0] = 61.184
    dt[yr < 1994.5] = 60.184; dt[yr < 1993.5] = 59.184; dt[yr < 1992.5] = 58.184; dt[yr < 1991.0] = 57.184
    dt[yr < 1990.0] = 56.184
    return dt / DAY_S

def gmst_rad(mjd_ut):
    T = (mjd_ut - 51544.5)
    g = 280.46061837 + 360.98564736629 * T
    return np.deg2rad(g % 360.0)

class Earth:
    def __init__(self, bsp):
        k = SPK.open(bsp)
        self.emb = k[0, 3]; self.e = k[3, 399]; self.sun = k[0, 10]
    def helio(self, mjd_tt):
        """heliocentric Earth position (AU) and velocity (AU/day), equatorial J2000"""
        jd = mjd_tt + 2400000.5
        p = np.zeros((3, len(jd))); v = np.zeros((3, len(jd)))
        for seg, sgn in ((self.emb, 1), (self.e, 1), (self.sun, -1)):
            pp, vv = seg.compute_and_differentiate(jd)
            p += sgn * pp; v += sgn * vv
        return (p / AU_KM).T, (v / AU_KM).T  # vv is km/day

def load_obs(path, codes, earth):
    t0 = time.time()
    raw = np.fromfile(path, dtype='S81')
    n = len(raw)
    A = raw.view('S1').reshape(n, 81)
    def col(a, b): return A[:, a-1:b].view(f'S{b-a+1}').ravel()
    note2 = col(15, 15)
    is_sat = note2 == b's'
    desig = col(1, 12)
    station = col(78, 80)
    datestr = col(16, 32)
    yr = col(16, 19).astype(int); mo = col(21, 22).astype(int)
    daystr = col(24, 32)
    day = np.array([float(x) if x.strip() else np.nan for x in daystr]) if False else np.char.strip(daystr.astype(str)).astype(float)
    mjd = mjd_from_ymd(yr, mo, day)
    # RA/Dec for normal lines
    good = ~is_sat
    def ffloat(s, default=0.0):
        s = np.char.strip(s.astype(str)); out = np.full(len(s), default); m = s != ''
        out[m] = s[m].astype(float); return out
    import pandas as pd
    rastr = pd.Series(col(33, 44).astype(str)); decstr = pd.Series(col(45, 56).astype(str))
    rx = rastr.str.extract(r'^\s*(\d+)\s+(\d+)\s+([\d.]+)')
    dx = decstr.str.extract(r'^\s*([+-]?)\s*(\d+)\s+(\d+)\s+([\d.]+)')
    rok = rx[0].notna().values & dx[1].notna().values
    ra = np.full(n, np.nan); dec = np.full(n, np.nan)
    ra[rok] = (rx[0][rok].astype(float) + rx[1][rok].astype(float) / 60 + rx[2][rok].astype(float) / 3600).values * 15.0
    dsign = np.where(dx[0].values == '-', -1.0, 1.0)
    dec[rok] = dsign[rok] * (dx[1][rok].astype(float) + dx[2][rok].astype(float) / 60 + dx[3][rok].astype(float) / 3600).values
    good = good & rok
    mag = ffloat(col(66, 70), np.nan)
    band = col(71, 71)
    print(f'parsed {n} lines in {time.time()-t0:.1f}s', flush=True)
    # observer geocentric position (AU, equatorial J2000-ish)
    geo = np.zeros((n, 3))
    lon = np.zeros(n); rc = np.zeros(n); rs = np.zeros(n); known = np.zeros(n, bool)
    ust, inv = np.unique(station, return_inverse=True)
    for i, c in enumerate(ust):
        cs = c.decode()
        if cs in codes and codes[cs][1] != 0.0:
            lon[inv == i], rc[inv == i], rs[inv == i] = codes[cs]; known[inv == i] = True
        elif cs in codes:  # space-based or roving (rc=rs=0)
            known[inv == i] = True
    theta = gmst_rad(mjd) + np.deg2rad(lon)
    geo[:, 0] = rc * np.cos(theta) * ERAD_AU; geo[:, 1] = rc * np.sin(theta) * ERAD_AU; geo[:, 2] = rs * ERAD_AU
    # satellite lines: attach to preceding 'S' line with same desig/date/station
    sat_idx = np.where(is_sat)[0]
    if len(sat_idx):
        unit = col(34, 34)[sat_idx]
        def sfloat(s):
            s = np.char.replace(np.char.replace(np.char.strip(s.astype(str)), '+ ', '+'), '- ', '-')
            s = np.char.replace(np.char.replace(s, '+ ', '+'), '- ', '-')
            out = np.full(len(s), np.nan)
            for ii, x in enumerate(s):
                try: out[ii] = float(x)
                except ValueError: pass
            return out
        sx = sfloat(col(35, 46)[sat_idx]); sy = sfloat(col(47, 58)[sat_idx]); sz = sfloat(col(59, 70)[sat_idx])
        sc = np.where(unit == b'1', 1.0 / AU_KM, 1.0)
        key_all = np.char.add(np.char.add(desig.astype(str), datestr.astype(str)), station.astype(str))
        keys_sat = key_all[sat_idx]
        S_idx = np.where(note2 == b'S')[0]
        d = {k: j for k, j in zip(key_all[S_idx], S_idx)}
        hit = 0
        for k, x, y, z, s in zip(keys_sat, sx, sy, sz, sc):
            j = d.get(k)
            if j is not None and np.isfinite(x + y + z):
                geo[j] = (x * s, y * s, z * s); hit += 1
        print(f'satellite lines {len(sat_idx)}, matched {hit}')
    keep = good & known & np.isfinite(mjd) & (mjd > 40000)
    mjd_tt = mjd + delta_t_days(mjd)
    print('computing Earth ephemeris...', flush=True)
    ep, ev = earth.helio(mjd_tt[keep])
    obs_pos = ep + geo[keep]
    # observer velocity: Earth velocity + rotation of geocentric vector
    omega = 2 * np.pi * 1.00273781191135448  # rad/day
    g = geo[keep]
    obs_vel = ev + omega * np.stack([-g[:, 1], g[:, 0], np.zeros(len(g))], axis=1)
    return dict(desig=desig[keep], station=station[keep], mjd=mjd[keep], ra=ra[keep], dec=dec[keep], mag=mag[keep],
                band=band[keep], obs_pos=obs_pos, obs_vel=obs_vel, line_idx=np.where(keep)[0])

def build_tracklets(o, max_gap_days=0.5, min_obs=2):
    t0 = time.time()
    key = np.char.add(o['desig'].astype(str), o['station'].astype(str))
    order = np.lexsort((o['mjd'], key))
    key = key[order]; mjd = o['mjd'][order]
    brk = np.ones(len(key), bool)
    brk[1:] = (key[1:] != key[:-1]) | (np.diff(mjd) > max_gap_days)
    tid = np.cumsum(brk) - 1
    ntr = tid[-1] + 1
    counts = np.bincount(tid)
    print(f'{ntr} tracklets; obs-count histogram: {np.bincount(counts)[:10]}', flush=True)
    ra = np.deg2rad(o['ra'][order]); dec = np.deg2rad(o['dec'][order])
    u = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)
    P = o['obs_pos'][order]; V = o['obs_vel'][order]; mag = o['mag'][order]
    # per-tracklet linear fits via sums (vectorized with bincount)
    tmean = np.bincount(tid, mjd) / counts
    dt = mjd - tmean[tid]
    Stt = np.bincount(tid, dt * dt)
    def linfit(y):  # y: (n,k) -> value at tmean and slope per day
        k = y.shape[1]
        a = np.stack([np.bincount(tid, y[:, j]) for j in range(k)], 1) / counts[:, None]
        b = np.stack([np.bincount(tid, dt * y[:, j]) for j in range(k)], 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            b = np.where(Stt[:, None] > 0, b / np.where(Stt[:, None] > 0, Stt[:, None], 1), 0.0)
        return a, b
    u0, udot = linfit(u)
    # residuals of unit vectors (arcsec rms)
    pred = u0[tid] + udot[tid] * dt[:, None]
    res = np.linalg.norm(u - pred, axis=1)
    rms = np.sqrt(np.bincount(tid, res ** 2) / counts) * 206264.8
    P0, Pdot = linfit(P)
    V0, _ = linfit(V)
    nrm = np.linalg.norm(u0, axis=1); u0 = u0 / nrm[:, None]
    udot = udot - u0 * np.sum(udot * u0, axis=1)[:, None]  # keep perpendicular
    arc = np.bincount(tid, mjd, ) * 0  # placeholder
    tmin = np.full(ntr, np.inf); np.minimum.at(tmin, tid, mjd)
    tmax = np.full(ntr, -np.inf); np.maximum.at(tmax, tid, mjd)
    magm = np.where(np.isfinite(mag), mag, 0); magc = np.bincount(tid, np.isfinite(mag).astype(float))
    with np.errstate(invalid='ignore', divide='ignore'):
        magmean = np.where(magc > 0, np.bincount(tid, magm) / np.where(magc > 0, magc, 1), np.nan)
    first = np.where(brk)[0]
    sel = counts >= min_obs
    print(f'tracklet build {time.time()-t0:.1f}s; keeping {sel.sum()} with >= {min_obs} obs', flush=True)
    return dict(t=tmean[sel], u=u0[sel], udot=udot[sel], obs_pos=P0[sel], obs_vel=V0[sel], nobs=counts[sel],
                arc_days=(tmax - tmin)[sel], rms_arcsec=rms[sel], mag=magmean[sel],
                desig=o['desig'][order][first][sel], station=o['station'][order][first][sel],
                first_line=o['line_idx'][order][first][sel], tid_of_obs=tid, obs_order=order, sel=sel)

if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    codes = parse_obscodes('/home/ubuntu/iso/data/ObsCodes.html')
    earth = Earth('/home/ubuntu/iso/data/de440s.bsp')
    o = load_obs(src, codes, earth)
    tr = build_tracklets(o)
    np.savez(out, **{k: v for k, v in tr.items()})
    print('saved', out)
