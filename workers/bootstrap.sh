#!/bin/bash
# ISO search worker bootstrap. Usage: bash bootstrap.sh <window> [<window> ...]
# Sets up /home/ubuntu/iso (or $HOME/iso), downloads public inputs, builds ITF tracklets, runs the assigned windows end-to-end.
set -euo pipefail
ROOT=/home/ubuntu/iso
if [ ! -w /home/ubuntu ] 2>/dev/null && [ "$HOME" != /home/ubuntu ]; then ROOT=$HOME/iso; fi
mkdir -p $ROOT/data/itf $ROOT/pipe
cp "$(dirname "$0")"/*.py $ROOT/pipe/
if [ "$ROOT" != /home/ubuntu/iso ]; then sed -i "s#/home/ubuntu/iso#$ROOT#g" $ROOT/pipe/*.py; fi
python3 -m pip install -q --user "numpy>=2,<3" "numba>=0.60" scipy pandas jplephem 2>&1 | tail -1 || true
cd $ROOT/data
[ -s ObsCodes.html ] || curl -sSL -o ObsCodes.html https://www.minorplanetcenter.net/iau/lists/ObsCodes.html
[ -s de440s.bsp ] || curl -sSL -o de440s.bsp https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp
[ -s itf.txt.gz ] || curl -sSL -o itf.txt.gz https://www.minorplanetcenter.net/iau/ITF/itf.txt.gz
[ -s itf.txt ] || gunzip -k itf.txt.gz
cd $ROOT/pipe
[ -s $ROOT/data/itf_tracklets.npz ] || python3 -u ingest.py $ROOT/data/itf.txt $ROOT/data/itf_tracklets.npz 2>&1 | tail -3
python3 -u itf_full.py --plan 2>&1 | grep -v "^loaded" | tail -60 > $ROOT/data/itf/plan.txt
for w in "$@"; do
  python3 -u itf_full.py $w $w 2>&1 | grep -v "^loaded" | tee -a $ROOT/data/itf/run_w$w.log
  python3 -u itf_fit.py $w 2>&1 | grep -v "^loaded" | tee -a $ROOT/data/itf/fit_w$w.log
done
echo BOOTSTRAP_DONE
