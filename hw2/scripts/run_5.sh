#!/bin/bash

# Tested parameters
#batch_sizes=(500 1000 2000)
#learning_rates=(0.005 0.01 0.02)

# Optimal parameters (with good convergence)
batch_sizes=(2000)
learning_rates=(0.005)

for b in ${batch_sizes[*]}; do
  for r in ${learning_rates[*]}; do

    # Delete previous directory for the same experiment (if it exists)
    rm -rf data/ip_b$b\_r$r\_InvertedPendulum-v2

    # Run experiment
    python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 \
      -l 2 -s 64 -b $b -lr $r -rtg -nt --exp_name ip_b$b\_r$r

    # Plot the results of the experiment
    python plot.py data/ip_b$b\_r$r\_InvertedPendulum-v2 --save_name ip_b$b\_r$r

  done
done
