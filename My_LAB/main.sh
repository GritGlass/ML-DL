#!/bin/bash

python ./trainer/QTSeg/train.py \
  --config configs/qtseg.yaml \
  --epochs 100 \
  --batch_size 16 \
  --valid_type test \
  --metric dice
