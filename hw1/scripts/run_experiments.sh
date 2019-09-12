#!/bin/bash
mkdir -p experiments
rm -rf ./experiments/*
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2; do
  python main.py --envname $e --algorithm behavioral_cloning --epochs 200 --eval_period 10
  python main.py --envname $e --algorithm dagger --epochs 4000 --eval_period 200
done
python utils/report.py
