#!/bin/sh
set -e

# GeSlerp: Slerp + Prompt(layer-level) for UIE
# Example: Rel(CoNLL04), you can replace uie_data config with other tasks

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python train.py -m src.task \
  -dc conf/mirror-multi-task-prompt_layer_level.yaml \
  -c conf/uie_data/wPretrain.yaml \
  -c conf/uie_data/uie_negins/rel_conll04.yaml \
  -a task_name=GeSlerp_Rel_CoNLL04_prompt_layer
