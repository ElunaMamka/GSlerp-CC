#!/bin/sh
set -e

# GeSlerp: PN Prompt for UIE
# Example: Rel(CoNLL04)

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python train.py -m src.task \
  -dc conf/mirror-multi-task-NegPrompt.yaml \
  -c conf/uie_data/wPretrain.yaml \
  -c conf/uie_data/uie_PNprompt/rel_conll04.yaml \
  -a task_name=GeSlerp_Rel_CoNLL04_prompt_PN \
  -a include_instructions=False \
  -a base_model_path=null
