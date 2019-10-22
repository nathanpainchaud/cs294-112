#!/bin/bash

# Set optimal parameters found by `run_8.sh` script
b=50000
r=0.02

# Delete previous directories for the same experiments (if they exist)
rm -rf hc_no_rtg_no_nn_b$b\_r$r\_HalfCheetah-v2
rm -rf hc_rtg_no_nn_b$b\_r$r\_HalfCheetah-v2
rm -rf hc_no_rtg_nn_b$b\_r$r\_HalfCheetah-v2
rm -rf hc_rtg_nn_b$b\_r$r\_HalfCheetah-v2

# Run experiments
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
  -l 2 -s 32 -b $b -lr $r -nt --exp_name hc_no_rtg_no_nn_b$b\_r$r
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
  -l 2 -s 32 -b $b -lr $r -rtg -nt --exp_name hc_rtg_no_nn_b$b\_r$r
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
  -l 2 -s 32 -b $b -lr $r --nn_baseline -nt --exp_name hc_no_rtg_nn_b$b\_r$r
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
  -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline -nt --exp_name hc_rtg_nn_b$b\_r$r

# Plot results
python plot.py \
  data/hc_no_rtg_no_nn_b$b\_r$r\_HalfCheetah-v2 \
  data/hc_rtg_no_nn_b$b\_r$r\_HalfCheetah-v2 \
  data/hc_no_rtg_nn_b$b\_r$r\_HalfCheetah-v2 \
  data/hc_rtg_nn_b$b\_r$r\_HalfCheetah-v2 \
  --save_name hc_d0.95_$b\_r$r
