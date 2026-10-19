"""Extract worker structured_output (summary/fit lines, CANDIDATE lists, problems) from saved devin_session_interact overflow dumps
into data/itf/workers/w<tag>.txt. Usage: pull_workers.py [content.txt ...] (default: all overflow dumps mentioning the workers)."""
import re, ast, glob, sys
files = sys.argv[1:] or glob.glob('/home/ubuntu/.devin-files/devin-remote-overflows-1000/*/content.txt')
for cf in files:
    txt = open(cf).read()
    if 'ISO search worker' not in txt: continue
    for sid, body in zip(re.findall(r'\nSession (\w+):\n', txt), re.split(r'\nSession \w+:\n', txt)[1:]):
        m = re.search(r'title: ISO search worker: ITF windows (\S+)', body)
        if not m: continue
        tag = m.group(1); out = []
        for key in ('summary_line', 'fit_line'):
            for mm in re.finditer(r"'%s': '((?:[^'\\]|\\.)*)'" % key, body):
                if mm.group(1): out.append(mm.group(1))
        for mm in re.finditer(r"'candidates': (\[.*?\])(?=, '|\})", body, re.S):
            try:
                for c in ast.literal_eval(mm.group(1)): out.append(c.strip())
            except Exception as e: print('parse fail', tag, e)
        pm = re.search(r"problems: (.*)", body)
        if pm: out.append('PROBLEMS: ' + pm.group(1)[:800])
        if len(out) >= 2:
            open(f'/home/ubuntu/iso/data/itf/workers/w{tag}.txt', 'w').write('\n'.join(dict.fromkeys(out)) + '\n'); print(tag, len(out), 'lines')
