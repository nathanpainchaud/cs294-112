#!/bin/bash

batch_sizes=(10000 30000 50000)
learning_rates=(0.005 0.01 0.02)

for b in ${batch_sizes[*]}; do
  for r in ${learning_rates[*]}; do

    # Delete previous directory for the same experiment (if it exists)
    rm -rf data/hc_rtg_nn_d0.9_b$b\_r$r\_HalfCheetah-v2

    # Run experiment
    python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 \
      -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline -nt --exp_name hc_rtg_nn_d0.9_b$b\_r$r

    # Plot the results of the experiment
    python plot.py data/hc_rtg_nn_d0.9_b$b\_r$r\_HalfCheetah-v2 --save_name hc_rtg_nn_d0.9_b$b\_r$r

  done

  # Plot the results of the experiment
  python plot.py \
    data/hc_rtg_nn_d0.9_b$b\_r0.005_HalfCheetah-v2 \
    data/hc_rtg_nn_d0.9_b$b\_r0.01_HalfCheetah-v2 \
    data/hc_rtg_nn_d0.9_b$b\_r0.02_HalfCheetah-v2 \
    --save_name hc_rtg_nn_d0.9_b$b

done
