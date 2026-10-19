"""find_orb screen of Stage-3 v2 candidates for a merged NSC+ITF tracklet file (ISO_TRK) and search dir (ISO_OUT).
Members with first_line >= 0 are ITF lines; first_line < 0 means -(1+index) into nsc_tracklets.npz (W84 detections)."""
import re, glob, os, subprocess, numpy as np, concurrent.futures as cf
from search import load_tracklets
from nsc_obs80 import _sexa, _date
D = '/home/ubuntu/iso/data'; TRK = os.environ['ISO_TRK']; OUT = os.environ['ISO_OUT']; CD = f'{OUT}/cand'; os.makedirs(CD, exist_ok=True)
tr = load_tracklets(TRK)
lines = open(f'{D}/itf.txt', encoding='latin-1').read().splitlines()
N = np.load(f'{D}/nsc/nsc_tracklets.npz'); NOBS, NORD, NFL, NN = N['obs'], N['obs_order'], N['first_line'], N['nobs']
def member_lines(i, tag):
    fl = int(tr['first_line'][i]); k = int(tr['nobs'][i])
    if fl >= 0: return ['     %-7s%s' % (tag, l[12:]) for l in lines[fl:fl + k]]
    j = -1 - fl; rows = NOBS[NORD[NFL[j]:NFL[j] + NN[j]]]; out = []
    for mjd, ra, dec, mag, _ in rows:
        r, dc = _sexa(ra, dec); m = f'{mag:5.1f} r' if np.isfinite(mag) and 0 < mag < 30 else '       '
        out.append((f'     {tag:<7s}  C{_date(mjd)}{r}{dc}'.ljust(65) + m).ljust(77) + 'W84')
    return out
cands = []
for f in sorted(glob.glob(f'{OUT}/fit2_w*.npy')):
    for k, x, te, rms, c2 in np.load(f, allow_pickle=True):
        idx = np.asarray(k, int); cands.append((os.path.basename(f), idx, f'rms {rms:.2f} chi2/dof {c2:.1f}'))
seen = set(); jobs = []
for src, idx, txt in cands:
    key = tuple(sorted(idx.tolist()))
    if key in seen or len(idx) < 2: continue
    seen.add(key); jobs.append(('NSC%04d' % len(seen), src, idx, txt))
def run(job):
    tag, src, idx, txt = job
    tt = tr['t'][idx]; st = sorted(set(tr['station'][idx].astype(str))); nn = int((tr['first_line'][idx] < 0).sum())
    obsf = f'{CD}/{tag}.obs'
    with open(obsf, 'w') as g:
        for i in idx: g.write('\n'.join(member_lines(i, tag)) + '\n')
    try:
        p = subprocess.Popen(['/home/ubuntu/iso/src/find_orb/fo', obsf, '-e', obsf + '.elem'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=CD, start_new_session=True)
        so, _ = p.communicate(timeout=600); o = re.sub(r'\x1b\[[0-9;]*m', '', so); res0 = None
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9); p.communicate(); o = ''; res0 = 'timeout'
    m = re.search(r'a=([\d.]+),\s*\S*e=([\d.]+)', o); q = re.search(r'q=([\d.]+),\s*e=([\d.]+)', o); used = re.search(r'(\d+) /\s*(\d+) obs', o)
    res = res0 or ((f'a={m.group(1)} e={m.group(2)}' if m else (f'q={q.group(1)} e={q.group(2)}' if q else 'no a/e parsed')) + (f' using {used.group(1)}/{used.group(2)} obs' if used else ''))
    return f'{tag} {src} trk {len(idx)} (nsc {nn}) MJD {tt.min():.1f}-{tt.max():.1f} ({tt.max()-tt.min():.0f} d) stations {st} {txt} -> find_orb {res}'
with open(f'{OUT}/fo_screen.txt', 'w') as out, cf.ThreadPoolExecutor(6) as ex:
    for r in cf.as_completed([ex.submit(run, j) for j in jobs]):
        out.write(r.result() + '\n'); out.flush(); print(r.result(), flush=True)
print(f'{len(jobs)} candidates screened')
