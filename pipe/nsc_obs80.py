"""Write NSC tracklet detections as MPC 80-column lines (station W84) for find_orb. Usage: lines = obs80(npz, tracklet_ids, tag)."""
import numpy as np
def _sexa(ra, dec):
    h = ra / 15.0; hh = int(h); mm = int((h - hh) * 60); ss = ((h - hh) * 60 - mm) * 60
    sgn = '-' if dec < 0 else '+'; d = abs(dec); dd = int(d); dm = int((d - dd) * 60); ds = ((d - dd) * 60 - dm) * 60
    return f'{hh:02d} {mm:02d} {ss:06.3f}', f'{sgn}{dd:02d} {dm:02d} {ds:05.2f}'
def _date(mjd):
    from datetime import datetime, timedelta
    d = datetime(1858, 11, 17) + timedelta(days=float(mjd))
    frac = mjd - np.floor(mjd)
    return f'{d.year:04d} {d.month:02d} {d.day + frac:09.6f}'
def obs80(d, tids, tag, station='W84'):
    out = []
    for t in tids:
        f = int(d['first_line'][t]); n = int(d['nobs'][t])
        for row in d['obs_table'][f:f + n]:
            mjd, ra, dec, mag = row[0], row[1], row[2], row[3]
            r, dc = _sexa(ra, dec)
            m = f'{mag:5.1f} r' if np.isfinite(mag) and 0 < mag < 30 else '       '
            out.append((f'     {tag:<7s}  C{_date(mjd)}{r}{dc}'.ljust(65) + m).ljust(77) + station)
    return out
if __name__ == '__main__':
    import sys
    d = np.load(sys.argv[1]); print('\n'.join(obs80(d, [0, 1], 'NSCTEST')))
