#!/bin/bash
# ISO tier-2a worker. Usage: bash bootstrap2a.sh <npz_path> <window> <pass_lo> <pass_hi>
# Runs itf_full.py (kernel + lossless pair hashing + per-pass triangle clustering) for passes pass_lo..pass_hi-1 of one window.
set -uo pipefail
NPZ=$1; W=$2; PLO=$3; PHI=$4
ROOT=/home/ubuntu/iso; [ -w /home/ubuntu ] || ROOT=$HOME/iso
mkdir -p $ROOT/pipe $ROOT/data/nsc/tier2a
cp "$(dirname "$0")"/*.py $ROOT/pipe/
[ "$ROOT" = /home/ubuntu/iso ] || sed -i "s#/home/ubuntu/iso#$ROOT#g" $ROOT/pipe/*.py
python3 -m pip install -q --user "numpy>=2,<3" "numba>=0.60" scipy 2>&1 | tail -1 || true
if ! swapon --show | grep -q swapfile; then sudo -n fallocate -l 32G /swapfile && sudo -n chmod 600 /swapfile && sudo -n mkswap /swapfile >/dev/null && sudo -n swapon /swapfile; fi
cp "$NPZ" $ROOT/data/nsc/tier2a_tracklets.npz
export ISO_TRK=$ROOT/data/nsc/tier2a_tracklets.npz ISO_OUT=$ROOT/data/nsc/tier2a ISO_STREAM=1 ISO_PASS_TRK=10000 ISO_PASS_RANGE=$PLO:$PHI
cd $ROOT/pipe
python3 -u itf_full.py --plan 2>&1 | tail -4
python3 -u itf_full.py $W $W
echo "RUN_EXIT $?"
cd $ROOT/data/nsc/tier2a && tar czf $ROOT/clusters_w0${W}_p${PLO}_${PHI}.tgz clusters_w0${W}_p*.pkl && ls -la $ROOT/clusters_w0${W}_p${PLO}_${PHI}.tgz
echo BOOTSTRAP_DONE
