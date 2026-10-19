import pandas as pd, numpy as np
e = pd.read_csv('/home/ubuntu/iso/data/nsc/exposures_c4d.csv')
e = e[np.isfinite(e.ra) & np.isfinite(e.dec) & (e.exptime >= 20)]
e['night'] = np.floor(e.mjd - 0.7).astype(int)          # CTIO: night boundary ~ local noon (16:48 UT)
# pointing bins ~1 deg (DECam FOV 2.2 deg): same bin & same night => revisit candidate
e['ib'] = np.floor(e.dec + 90).astype(int); e['jb'] = np.floor(e.ra * np.cos(np.radians(e.dec))).astype(int)
grp = e.groupby(['night', 'ib', 'jb'])
agg = grp.agg(n=('mjd', 'size'), t0=('mjd', 'min'), t1=('mjd', 'max'), nmeas=('nmeas', 'sum'))
rev = agg[(agg.n >= 2) & ((agg.t1 - agg.t0) * 1440 >= 10) & ((agg.t1 - agg.t0) * 24 <= 5)]
print(f'exposures {len(e)}, pointing-night groups {len(agg)}, revisit groups (>=2 exp, 10 min..5 h apart) {len(rev)}')
print(f'exposures in revisit groups {rev.n.sum()}, measurements {rev.nmeas.sum()/1e9:.1f} B, median group size {rev.n.median()}')
print('per-year revisit groups:'); yr = (rev.reset_index().night - 51544) / 365.25 + 2000; print(np.floor(yr).value_counts().sort_index().to_string())
rev.reset_index().to_csv('/home/ubuntu/iso/data/nsc/revisit_groups.csv', index=False)
ids = e.set_index(['night', 'ib', 'jb']).loc[rev.index, ['exposure', 'expnum', 'mjd', 'ra', 'dec', 'filter', 'exptime', 'nmeas']].reset_index()
ids.to_csv('/home/ubuntu/iso/data/nsc/revisit_exposures.csv', index=False); print('saved', len(ids), 'exposures')
