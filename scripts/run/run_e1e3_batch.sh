#!/bin/bash
PY=/root/miniconda3/envs/sgs/bin/python
LOG=/root/autodl-tmp/e1e3_batch.log
: > $LOG
for seq in 360 desk2 floor plant rpy teddy; do
  echo "=== $seq start $(date +%H:%M:%S) ===" >> $LOG
  $PY /root/autodl-tmp/gen_e1_e3.py --seq $seq >> $LOG 2>&1
  echo "=== $seq done $(date +%H:%M:%S) ===" >> $LOG
done
echo "=== ALL DONE $(date +%H:%M:%S) ===" >> $LOG
