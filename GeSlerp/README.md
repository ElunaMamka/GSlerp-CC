# GeSlerp

这是从 `DarkMirror` 中抽取的“**Slerp + Prompt** 完成 UIE 任务”的关键代码最小集合。
目标是保留可训练、可评估、可复现的核心链路，不包含大体量数据和无关模块。

## 1. 代码结构（关键部分）

- `train.py`：训练入口（`rex` 注册任务）
- `eval_one.py`：单 checkpoint 评估入口
- `src/task.py`：UIE 任务定义、训练/评估流程
- `src/transform.py`：UIE 样本到 prompt 输入的转换
- `src/model.py`：Prompt 模型（含 Event/PN/Layer-Level 变体）
- `src/base_models.py`：底层 Prompt Fusion 模型，含 `slerp_for_batch*` 实现
- `src/metric.py`：UIE 指标
- `src/utils.py`：解码工具
- `conf/mirror-multi-task-prompt_layer_level.yaml`：Slerp+Prompt Layer 训练配置
- `conf/mirror-multi-task-NegPrompt.yaml`：PN-Prompt 训练配置
- `conf/uie_data/*`：UIE 数据任务配置（保留了常用示例）
- `scripts/*.sh`：可直接运行的训练/评估脚本

## 2. 你关心的 Slerp 与 Prompt 在哪里

- Slerp 核心函数：
  - `src/base_models.py`
    - `slerp_for_batch_hidden(...)`
    - `slerp_for_batch(...)`
- Prompt 注入/替换（UIE）：
  - `src/transform.py`
    - `CachedLabelPointerTransformWithPromptReplace`
    - `CachedLabelPointerTransformWith_PN_PromptReplace`
  - `src/model.py`
    - `SchemaGuidedInstructBertModelWithPromptReplace_Event`
    - `SchemaGuidedInstructBertModelWithPromptReplace_PN_Event`
    - `SchemaGuidedInstructBertModelWithPromptReplace_Event_Layer_Level`
  - `src/task.py`
    - `SchemaGuidedInstructBertTaskWithPromptReplace`
    - `SchemaGuidedInstructBertTaskWith_PN_PromptReplace`
    - `SchemaGuidedInstructBertTaskWithPromptReplace_Layer_Level`

## 3. 环境与依赖

### 3.1 Python 依赖

```bash
pip install -r requirments.txt
```

### 3.2 REx 框架

本代码依赖 `rex`（`from rex...`）。请确保你环境里有 REx（例如安装你原项目中的 `RExmain`）。

### 3.3 模型与数据路径

- 默认预训练模型目录：`deberta-v3-large`
- 默认数据路径仍沿用 `DarkMirror` 的 `resources/Mirror/...`
- 你只复制了关键代码，数据集与大模型权重需自行准备

## 4. 训练

### 4.1 Slerp + Prompt Layer（推荐主线）

```bash
sh scripts/run_slerp_prompt_layer.sh
```

对应命令（Rel-CoNLL04 示例）：

```bash
python train.py -m src.task \
  -dc conf/mirror-multi-task-prompt_layer_level.yaml \
  -c conf/uie_data/wPretrain.yaml \
  -c conf/uie_data/uie_negins/rel_conll04.yaml \
  -a task_name=GeSlerp_Rel_CoNLL04_prompt_layer
```

### 4.2 PN Prompt

```bash
sh scripts/run_pn_prompt.sh
```

## 5. 评估

### 5.1 单 checkpoint 评估

```bash
sh scripts/eval_checkpoint.sh <task_dir> <ckpt_path>
```

例如：

```bash
sh scripts/eval_checkpoint.sh \
  mirror_outputs/GeSlerp_Rel_CoNLL04_prompt_layer \
  mirror_outputs/GeSlerp_Rel_CoNLL04_prompt_layer/ckpt/SchemaGuidedInstructBertModel.best.pth
```

`eval_one.py` 会按 `data_pairs` 里的数据集路径评估，你可以按任务修改该列表。

## 6. 说明

- 代码主体尽量保持原样，仅做“关键代码抽取 + 可运行脚本 + 文档”整理。
- 如果你要扩展到更多 UIE 子任务，可继续从 `conf/uie_data` 增加对应 yaml。
