#!/bin/sh
set -e

# Usage:
# sh scripts/eval_checkpoint.sh <task_dir> <ckpt_path>
# Example:
# sh scripts/eval_checkpoint.sh mirror_outputs/GeSlerp_Rel_CoNLL04_prompt_layer mirror_outputs/GeSlerp_Rel_CoNLL04_prompt_layer/ckpt/SchemaGuidedInstructBertModel.best.pth

TASK_DIR=$1
CKPT=$2

if [ -z "$TASK_DIR" ] || [ -z "$CKPT" ]; then
  echo "Usage: sh scripts/eval_checkpoint.sh <task_dir> <ckpt_path>"
  exit 1
fi

python eval_one.py --task_dir "$TASK_DIR" --ckpt "$CKPT"
