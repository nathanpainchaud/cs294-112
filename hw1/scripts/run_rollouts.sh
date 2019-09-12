#!/bin/bash
ROLLOUTS=${1:-20}
mkdir -p expert_data
rm -rf ./expert_data/*
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python run_expert.py experts/$e.pkl $e --num_rollouts="$ROLLOUTS"
done