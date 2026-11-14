#!/bin/bash
# Resumable launcher: safe to run again after any restart; finished chunks are skipped.
cd ./std
pgrep -f "inc_calib.py 0$" >/dev/null || setsid nohup python3 inc_calib.py 0 >> ckpt/out_calib0.txt 2>&1 < /dev/null &
pgrep -f "inc_calib.py 0.3" >/dev/null || setsid nohup python3 inc_calib.py 0.3 >> ckpt/out_calib03.txt 2>&1 < /dev/null &
pgrep -f "inc_robust.py" >/dev/null || setsid nohup python3 inc_robust.py >> ckpt/out_robust.txt 2>&1 < /dev/null &
pgrep -f "inc_W1.py" >/dev/null || setsid nohup nice -n 10 python3 inc_W1.py 0 1 2 3 4 5 >> ckpt/out_W1.txt 2>&1 < /dev/null &
