"""find_orb screen of Stage-3 v2 candidates. Local windows: exact member indices from fit2_wNN.npy. Worker windows: CANDIDATE2
lines in workers/w*.txt or workers/fit2_w*.log, matched by (designation, MJD range) in the local ITF snapshot (>=8 desigs listed).
Output: data/cand2/fo_screen2.txt"""
import re, glob, os, subprocess, numpy as np
from search import load_tracklets
D = '/home/ubuntu/iso/data'; os.makedirs(f'{D}/cand2', exist_ok=True); tr = load_tracklets(f'{D}/itf_tracklets.npz')
desig = np.char.strip(tr['desig'].astype(str))
lines = open(f'{D}/itf.txt', encoding='latin-1').read().splitlines()
cands = []   # (src, n, idx array, text)
for f in sorted(glob.glob(f'{D}/itf/fit2_w*.npy')):
    for k, x, te, rms, c2 in np.load(f, allow_pickle=True):
        idx = np.asarray(k, int); cands.append((os.path.basename(f), len(idx), idx, f'rms {rms:.2f} chi2/dof {c2:.1f}'))
for f in glob.glob(f'{D}/itf/workers/*.txt') + glob.glob(f'{D}/itf/workers/fit2_w*.log'):
    for l in open(f, errors='replace'):
        if 'CANDIDATE2' not in l or 'desigs' not in l: continue
        ds = [d.strip() for d in re.findall(r"np\.str_\(\\?'([^'\\]*)\\?'\)", l.split('desigs')[1])]
        mm_ = re.search(r"CANDIDATE2: (\d+) tracklets", l); mj = re.search(r"MJD ([\d.]+)-([\d.]+)", l)
        if not mm_ or not mj or not ds: continue
        idx = np.flatnonzero(np.isin(desig, ds) & (tr['t'] >= float(mj.group(1)) - 0.5) & (tr['t'] <= float(mj.group(2)) + 0.5))
        cands.append((os.path.basename(f), int(mm_.group(1)), idx, l.strip()[:100]))

import concurrent.futures as cf
prev = {}
if os.path.exists(f'{D}/cand2/fo_screen2.txt'):
    for l in open(f'{D}/cand2/fo_screen2.txt'):
        m_ = re.search(r' local_trk (\d+) MJD ([\d.]+)-([\d.]+)', l)
        if m_: prev[(l.split()[1], m_.group(1), m_.group(2), m_.group(3))] = l.rstrip('\n')
seen = set(); jobs = []; nt = 0
for src, n, idx, txt in cands:
    key = tuple(sorted(idx.tolist()))
    if key in seen or len(idx) < 2: continue
    seen.add(key); nt += 1; jobs.append(('ISO%04d' % nt, src, n, idx, txt))
def run(job):
    tag, src, n, idx, txt = job
    tt = tr['t'][idx]; st = sorted(set(tr['station'][idx].astype(str)))
    pk = (src, str(len(idx)), f'{tt.min():.1f}', f'{tt.max():.1f}')
    if pk in prev: return prev[pk]
    obsf = f'{D}/cand2/{tag}.obs'
    with open(obsf, 'w') as g:
        for i in idx:
            fl = int(tr['first_line'][i]); k = int(tr['nobs'][i])
            for l in lines[fl:fl + k]: g.write('     %-7s%s\n' % (tag, l[12:]))
    try:
        p = subprocess.Popen(['/home/ubuntu/iso/src/find_orb/fo', obsf, '-e', obsf + '.elem'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, cwd=f'{D}/cand2', start_new_session=True)
        so, _ = p.communicate(timeout=600)
        o = re.sub(r'\x1b\[[0-9;]*m', '', so)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9); p.communicate(); o = ''; res0 = 'timeout'
    else: res0 = None
    m = re.search(r'a=([\d.]+),\s*\S*e=([\d.]+)', o); q = re.search(r'q=([\d.]+),\s*e=([\d.]+)', o); used = re.search(r'(\d+) /\s*(\d+) obs', o)
    res = res0 or ((f'a={m.group(1)} e={m.group(2)}' if m else (f'q={q.group(1)} e={q.group(2)}' if q else 'no a/e parsed')) + (f' using {used.group(1)}/{used.group(2)} obs' if used else ''))
    return f'{tag} {src:14s} n_listed {n} local_trk {len(idx)} MJD {tt.min():.1f}-{tt.max():.1f} ({tt.max()-tt.min():.0f} d) stations {st[:6]} -> find_orb {res} | {txt}'
out = open(f'{D}/cand2/fo_screen2.txt', 'w')
with cf.ThreadPoolExecutor(6) as ex:
    for fut in cf.as_completed([ex.submit(run, j) for j in jobs]):
        line = fut.result(); print(line, flush=True); out.write(line + '\n'); out.flush()
out.close(); print(f'{nt} candidates screened')
