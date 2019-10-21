#!/bin/bash

# Delete previous directories for the same experiments (if they exist)
rm -rf sb_no_rtg_dna_CartPole-v0
rm -rf sb_rtg_dna_CartPole-v0
rm -rf sb_rtg_na_CartPole-v0
rm -rf lb_no_rtg_dna_CartPole-v0
rm -rf lb_rtg_dna_CartPole-v0
rm -rf lb_rtg_na_CartPole-v0

# Small batch runs
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna -nt --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna -nt --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -nt --exp_name sb_rtg_na

# Large batch runs
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna -nt --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna -nt --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -nt --exp_name lb_rtg_na

# Plot results
python plot.py data/sb_no_rtg_dna_CartPole-v0 data/sb_rtg_dna_CartPole-v0 data/sb_rtg_na_CartPole-v0 --save_name cp_sb
python plot.py data/lb_no_rtg_dna_CartPole-v0 data/lb_rtg_dna_CartPole-v0 data/lb_rtg_na_CartPole-v0 --save_name cp_lb
