"""Pull NSC DR2 measurements of weakly-detected objects (ndet<=2: never matched to a persistent source) for every exposure in a
same-night revisit group. Batched IN(...) queries (10 exposures: 13 s vs 4.5 s each singly). Resumable: one gzip CSV per batch,
exposures already present in meas/ (from the single-exposure fetcher) are skipped. Network-bound."""
import pandas as pd, numpy as np, subprocess, os, sys, time, gzip, glob, concurrent.futures as cf
OUT = '/home/ubuntu/iso/data/nsc/meas'
ex = pd.read_csv('/home/ubuntu/iso/data/nsc/revisit_exposures.csv').sort_values('mjd', ascending=False)
have = {os.path.basename(f)[:-7] for f in glob.glob(f'{OUT}/c4d_*.csv.gz')}
done_b = set()
for f in glob.glob(f'{OUT}/batch_*.done'):
    done_b.update(open(f).read().split())
todo = [e for e in ex.exposure.tolist() if e not in have and e not in done_b]
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
batches = [todo[i:i + BS] for i in range(0, len(todo), BS)]
print(f'{len(have)+len(done_b)} exposures already fetched, {len(todo)} to go in {len(batches)} batches of {BS}', flush=True)
Q = ("SELECT m.measid,m.objectid,m.exposure,m.ra,m.dec,m.raerr,m.decerr,m.mjd,m.mag_auto,m.magerr_auto,m.class_star,m.fwhm,m.flags,o.ndet "
     "FROM nsc_dr2.meas m JOIN nsc_dr2.object o ON o.id=m.objectid WHERE m.exposure IN ({}) AND o.ndet<=2")
def fetch(ib):
    b = batches[ib]; tag = f'{OUT}/batch_{b[0]}'
    for att in range(4):
        r = subprocess.run(['curl', '-s', '-m', '900', '-G', 'https://datalab.noirlab.edu/tap/sync', '--data-urlencode', 'REQUEST=doQuery', '--data-urlencode', 'LANG=ADQL', '--data-urlencode', 'FORMAT=csv', '--data-urlencode', 'QUERY=' + Q.format(','.join(f"'{e}'" for e in b))], capture_output=True)
        out = r.stdout
        if r.returncode == 0 and out.startswith(b'measid'):
            with gzip.open(tag + '.csv.gz.tmp', 'wb') as g: g.write(out)
            os.rename(tag + '.csv.gz.tmp', tag + '.csv.gz'); open(tag + '.done', 'w').write('\n'.join(b))
            return len(b), out.count(b'\n') - 1
        time.sleep(10 * (att + 1))
    return 0, -1
nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
t0 = time.time(); done = 0; rows = 0; fail = 0; nb = 0
with cf.ThreadPoolExecutor(nw) as pool:
    for n, nr in pool.map(fetch, range(len(batches))):
        done += n; nb += 1
        if nr < 0: fail += 1
        else: rows += nr
        if nb % 10 == 0: print(f'{done}/{len(todo)} exposures, {rows} rows, {fail} failed batches, {time.time()-t0:.0f}s ({done/max(time.time()-t0,1)*60:.0f} exp/min)', flush=True)
print('finished', done, rows, fail)
