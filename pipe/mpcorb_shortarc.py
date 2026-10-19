import gzip, numpy as np, collections
n = 0; rows = []
with gzip.open('/home/ubuntu/iso/data/MPCORB.DAT.gz', 'rt', errors='replace') as f:
    started = False
    for line in f:
        if not started:
            if line.startswith('----'): started = True
            continue
        if len(line) < 160: continue
        try:
            e = float(line[70:79]); a = float(line[92:103]); nobs = int(line[117:122]); nopp = int(line[123:126])
        except ValueError: continue
        arc = line[127:136].strip()
        if nopp == 1:
            try: arcd = float(arc.split()[0]) if 'days' in arc else np.nan
            except ValueError: arcd = np.nan
        else: arcd = np.nan
        rows.append((line[0:7], e, a, nobs, nopp, arcd, line[166:194].strip()))
print('total objects', len(rows))
r = np.array([(x[3], x[4], x[5], x[1]) for x in rows])
single = r[:, 1] == 1
print('single-opposition', single.sum())
for d in (3, 7, 14, 30, 60):
    sel = single & (r[:, 2] <= d); print(f'  arc <= {d:3d} d: {sel.sum():7d} objects, median nobs {np.median(r[sel, 0]) if sel.sum() else 0:.0f}')
sel = single & (r[:, 2] <= 60) & np.isfinite(r[:, 2])
import pickle; pickle.dump([x for x, s in zip(rows, sel) if s], open('/home/ubuntu/iso/data/mpcorb_shortarc.pkl', 'wb'))
print('saved short-arc (<=60 d) list:', sel.sum())
