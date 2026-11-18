#!/bin/sh
# Runs the step 2b training in the background with a log file.
#
#   scripts/train_2b.sh            start a run, or resume it if a last checkpoint exists
#   tail -f build/neural/train_2b.log
#
# Extra arguments go to neural.train, e.g. scripts/train_2b.sh --steps 60000
set -eu
cd "$(dirname "$0")/.."
mkdir -p build/neural
nohup .venv/bin/python -m neural.train --resume "$@" > build/neural/train_2b.log 2>&1 &
echo "training pid $! -> build/neural/train_2b.log"
