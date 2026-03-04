import math
import os
import json
import re
from collections import defaultdict
from datetime import datetime
from typing import List
from typing import Optional, Union
from pathlib import Path
import torch
import torch.optim as optim
from rex import accelerator
from rex.data.data_manager import DataManager
from rex.data.dataset import CachedDataset, StreamReadDataset
from rex.tasks.simple_metric_task import SimpleMetricTask
from rex.utils.batch import decompose_batch_into_instances
from rex.utils.logging import logger
from rex.utils.config import ConfigParser
from rex.utils.dict import flatten_dict
from rex.utils.progress_bar import pbar
from rex.utils.io import load_jsonlines
from rex.metrics import safe_division
from rex.utils.registry import register

from omegaconf import DictConfig, OmegaConf

from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from rex.utils.vars import (
    CONFIG_PARAMS_FILENAME,
)

from .metric import MrcNERMetric, MrcSpanMetric, MultiPartSpanMetric
from .model import (
    MrcGlobalPointerModel,
    MrcPointerMatrixModel,
    SchemaGuidedInstructBertModel,
    SchemaGuidedInstructBertModelWithPromptReplace,
    SchemaGuidedInstructBertModelWithPromptReplace_Concatenate,
    SchemaGuidedInstructBertModelWithPromptReplace_Concatenate_GNN,
    SchemaGuidedInstructBertModelWithPromptReplace_Event,
    SchemaGuidedInstructBertModelWithPromptReplace_PN_Event,
    SchemaGuidedInstructBertModelWithPromptReplace_Event_Layer_Level,
)
from .config import Prompt_Fusion_Config
from .transform import (
    CachedLabelPointerTransform,
    CachedPointerMRCTransform,
    CachedPointerTaggingTransform,
    CachedLabelPointerTransformWithPromptReplace,
    CachedLabelPointerTransformWith_PN_PromptReplace
)

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datetime import datetime

def from_local_pretrained(model,state_dict):
    if 'prompt_embeddings ' in state_dict:
        model.load_state_dict(state_dict)
    else :
        new_state_dict = {}
        for key in state_dict:
            # new_key = key.replace('deberta.', '')
            # 映射嵌入层权重
            if 'plm.embeddings.word_embeddings' in key:
                new_state_dict[key] = state_dict[key]
                # prompt_embedding_key = key.replace('deberta.', '') 
                # prompt_embedding_key = prompt_embedding_key.replace('embeddings.', 'prompt_embeddings.') 
                new_state_dict['plm.prompt_embeddings.word_embeddings.weight'] = state_dict[key]
            elif 'plm.embeddings.position_embeddings' in key:
                if model.position_biased_input:
                # 如果你的模型使用位置嵌入，你需要确保你的模型中有相应的参数接收这些权重
                    new_state_dict[key] = state_dict[key]
                    prompt_embedding_key = key.replace('embeddings.', 'prompt_embeddings.') 
                    new_state_dict[prompt_embedding_key] = state_dict[key]
                else:
                    continue  # 此例中忽略，因为你的模型没有明确这一点
            elif 'plm.embeddings.LayerNorm' in key:
                # new_key = new_key.replace('embeddings.', 'embeddings.LayerNorm.')
                new_state_dict[key] = state_dict[key]
                # prompt_embedding_key = key.replace('deberta.', '') 
                prompt_embedding_key = key.replace('embeddings.', 'prompt_embeddings.') 
                new_state_dict[prompt_embedding_key] = state_dict[key]
            
            # 映射编码器层权重
            elif 'plm.encoder.layer.' in key:
                # new_key = key.replace('deberta.','')
                layer_match = re.match(r"plm.encoder\.layer\.(\d+)\.", key)
                if layer_match:
                    layer_id = int(layer_match.group(1))
                    # 映射到常规层或提示层
                    if layer_id < model.plm.config.num_hidden_prompt_layers:
                        prompt_key = key.replace('encoder.layer','encoder_prompt.layer')
                        new_state_dict[prompt_key] = state_dict[key]
                    main_key = key.replace('encoder.layer','encoder_main.layer')
                    new_state_dict[main_key] = state_dict[key]

            elif 'plm.encoder.' in key:
                prompt_key = key.replace('encoder.','encoder_prompt.')
                main_key = key.replace('encoder.','encoder_main.')
                new_state_dict[prompt_key] = state_dict[key]
                new_state_dict[main_key] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]  
                    


        # 使用新的state_dict来加载权重
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    return model



@register("task")
class MrcTaggingTask(SimpleMetricTask):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)

    def after_initialization(self):
        now_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.tb_logger: SummaryWriter = SummaryWriter(
            log_dir=self.task_path / "tb_summary" / now_string,
            comment=self.config.comment,
        )

    def after_whole_train(self):
        self.tb_logger.close()

    def get_grad_norm(self):
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         grads = param.grad.detach().data
        #         grad_norm = (grads.norm(p=2) / grads.numel()).item()
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def log_loss(
        self, idx: int, loss_item: float, step_or_epoch: str, dataset_name: str
    ):
        self.tb_logger.add_scalar(
            f"loss/{dataset_name}/{step_or_epoch}", loss_item, idx
        )
        # self.tb_logger.add_scalars(
        #     "lr",
        #     {
        #         str(i): self.optimizer.param_groups[i]["lr"]
        #         for i in range(len(self.optimizer.param_groups))
        #     },
        #     idx,
        # )
        self.tb_logger.add_scalar("lr", self.optimizer.param_groups[0]["lr"], idx)
        self.tb_logger.add_scalar("grad_norm_total", self.get_grad_norm(), idx)

    def log_metrics(
        self, idx: int, metrics: dict, step_or_epoch: str, dataset_name: str
    ):
        metrics = flatten_dict(metrics)
        self.tb_logger.add_scalars(f"{dataset_name}/{step_or_epoch}", metrics, idx)

    def init_transform(self):
        return CachedPointerTaggingTransform(
            self.config.max_seq_len,
            self.config.plm_dir,
            self.config.ent_type2query_filepath,
            mode=self.config.mode,
            negative_sample_prob=self.config.negative_sample_prob,
        )

    def init_data_manager(self):
        return DataManager(
            self.config.train_filepath,
            self.config.dev_filepath,
            self.config.test_filepath,
            CachedDataset,
            self.transform,
            load_jsonlines,
            self.config.train_batch_size,
            self.config.eval_batch_size,
            self.transform.collate_fn,
            use_stream_transform=False,
            debug_mode=self.config.debug_mode,
            dump_cache_dir=self.config.dump_cache_dir,
            regenerate_cache=self.config.regenerate_cache,
        )

    def init_model(self):
        # m = MrcPointerMatrixModel(
        m = MrcGlobalPointerModel(
            self.config.plm_dir,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
            mode=self.config.mode,
        )
        return m

    def init_metric(self):
        return MrcNERMetric()

    def init_optimizer(self):
        no_decay = r"(embedding|LayerNorm|\.bias$)"
        plm_lr = r"^plm\."
        non_trainable = r"^plm\.(emb|encoder\.layer\.[0-3])"

        param_groups = []
        for name, param in self.model.named_parameters():
            lr = self.config.learning_rate
            weight_decay = self.config.weight_decay
            if re.search(non_trainable, name):
                param.requires_grad = False
            if not re.search(plm_lr, name):
                lr = self.config.other_learning_rate
            if re.search(no_decay, name):
                weight_decay = 0.0
            param_groups.append(
                {"params": param, "lr": lr, "weight_decay": weight_decay}
            )
        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    def init_lr_scheduler(self):
        num_training_steps = int(
            len(self.data_manager.train_loader)
            * self.config.num_epochs
            * accelerator.num_processes
        )
        num_warmup_steps = math.floor(
            num_training_steps * self.config.warmup_proportion
        )
        # return get_linear_schedule_with_warmup(
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def predict_api(self, texts: List[str], **kwargs):
        raw_dataset = self.transform.predict_transform(texts)
        text_ids = sorted(list({ins["id"] for ins in raw_dataset}))
        loader = self.data_manager.prepare_loader(raw_dataset)
        # to prepare input device
        loader = accelerator.prepare_data_loader(loader)
        id2ents = defaultdict(set)
        for batch in loader:
            batch_out = self.model(**batch, is_eval=True)
            for _id, _pred in zip(batch["id"], batch_out["pred"]):
                id2ents[_id].update(_pred)
        results = [id2ents[_id] for _id in text_ids]

        return results


@register("task")
class MrcQaTask(MrcTaggingTask):
    def init_transform(self):
        return CachedPointerMRCTransform(
            self.config.max_seq_len,
            self.config.plm_dir,
            mode=self.config.mode,
        )

    def init_model(self):
        # m = MrcPointerMatrixModel(
        m = MrcGlobalPointerModel(
            self.config.plm_dir,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
            mode=self.config.mode,
        )
        return m

    def init_metric(self):
        return MrcSpanMetric()

    def predict_api(self, data: list[dict], **kwargs):
        """
        Args:
            data: a list of dict with query, context, and background strings
        """
        raw_dataset = self.transform.predict_transform(data)
        loader = self.data_manager.prepare_loader(raw_dataset)
        results = []
        for batch in loader:
            batch_out = self.model(**batch, is_eval=True)
            batch["pred"] = batch_out["pred"]
            instances = decompose_batch_into_instances(batch)
            for ins in instances:
                preds = ins["pred"]
                ins_results = []
                for index_list in preds:
                    ins_result = []
                    for i in index_list:
                        ins_result.append(ins["raw_tokens"][i])
                    ins_results.append(("".join(ins_result), tuple(index_list)))
                results.append(ins_results)

        return results


class StreamReadDatasetWithLen(StreamReadDataset):
    def __len__(self):
        return 631346


@register("task")
class SchemaGuidedInstructBertTask(MrcTaggingTask):
    # def __init__(self, config, **kwargs) -> None:
    #     super().__init__(config, **kwargs)

    #     from watchmen import ClientMode, WatchClient

    #     client = WatchClient(
    #         id=config.task_name,
    #         gpus=[4],
    #         req_gpu_num=1,
    #         mode=ClientMode.SCHEDULE,
    #         server_host="127.0.0.1",
    #         server_port=62333,
    #     )
    #     client.wait()

    # def init_lr_scheduler(self):
    #     num_training_steps = int(
    #         631346 / self.config.train_batch_size
    #         * self.config.num_epochs
    #         * accelerator.num_processes
    #     )
    #     num_warmup_steps = math.floor(
    #         num_training_steps * self.config.warmup_proportion
    #     )
    #     # return get_linear_schedule_with_warmup(
    #     return get_cosine_schedule_with_warmup(
    #         self.optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps,
    #     )

    def init_transform(self):
        self.transform: CachedLabelPointerTransform
        return CachedLabelPointerTransform(
            self.config.max_seq_len,
            self.config.plm_dir,
            mode=self.config.mode,
            label_span=self.config.label_span,
            include_instructions=self.config.get("include_instructions", True),
        )

    def init_data_manager(self):
        if self.config.get("stream_mode", False):
            DatasetClass = StreamReadDatasetWithLen
            transform = self.transform.transform
        else:
            DatasetClass = CachedDataset
            transform = self.transform
        return DataManager(
            self.config.train_filepath,
            self.config.dev_filepath,
            self.config.test_filepath,
            DatasetClass,
            transform,
            load_jsonlines,
            self.config.train_batch_size,
            self.config.eval_batch_size,
            self.transform.collate_fn,
            use_stream_transform=self.config.get("stream_mode", False),
            debug_mode=self.config.debug_mode,
            dump_cache_dir=self.config.dump_cache_dir,
            regenerate_cache=self.config.regenerate_cache,
        )

    def init_model(self):
        self.model = SchemaGuidedInstructBertModel(
            self.config.plm_dir,
            vocab_size=len(self.transform.tokenizer),
            use_rope=self.config.use_rope,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
        )
        print(self.config)
        if self.config.get("base_model_path"):
            self.load(
                self.config.base_model_path,
                load_config=False,
                load_model=True,
                load_optimizer=False,
                load_history=False,
            )
        return self.model

    def init_optimizer(self):
        no_decay = r"(embedding|LayerNorm|\.bias$)"
        plm_lr = r"^plm\."
        # non_trainable = r"^plm\.(emb|encoder\.layer\.[0-3])"
        non_trainable = "no_non_trainable"

        param_groups = []
        for name, param in self.model.named_parameters():
            lr = self.config.learning_rate
            weight_decay = self.config.weight_decay
            if re.search(non_trainable, name):
                param.requires_grad = False
            if not re.search(plm_lr, name):
                lr = self.config.other_learning_rate
            if re.search(no_decay, name):
                weight_decay = 0.0
            param_groups.append(
                {"params": param, "lr": lr, "weight_decay": weight_decay}
            )
        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    def init_metric(self):
        return MultiPartSpanMetric()

    def _convert_span_to_string(self, span, token_ids, tokenizer):
        string = ""
        if len(span) == 0 or len(span) > 2:
            pass
        elif len(span) == 1:
            string = tokenizer.decode(token_ids[span[0]])
        elif len(span) == 2:
            string = tokenizer.decode(token_ids[span[0] : span[1] + 1])
        return (string, self.reset_position(token_ids, span))

    def reset_position(self, token_ids: list[int], span: list[int]) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            input_ids = token_ids.cpu().tolist()
        if len(span) < 1:
            return span

        tp_token_id, tl_token_id = self.transform.tokenizer.convert_tokens_to_ids(
            [self.transform.tp_token, self.transform.tl_token]
        )
        offset = 0
        if tp_token_id in input_ids:
            offset = input_ids.index(tp_token_id) + 1
        elif tl_token_id in input_ids:
            offset = input_ids.index(tl_token_id) + 1
        return [i - offset for i in span]

    def predict_api(self, data: list[dict], **kwargs):
        """
        Args:
            data: a list of dict in UDI:
                {
                    "id": str,
                    "instruction": str,
                    "schema": {
                        "ent": list,
                        "rel": list,
                        "event": dict,
                        "cls": list,
                        "discontinuous_ent": list,
                        "hyper_rel": dict
                    },
                    "text": str,
                    "bg": str,
                    "ans": {},  # empty dict
                }
        """
        raw_dataset = [self.transform.transform(d) for d in data]
        loader = self.data_manager.prepare_loader(raw_dataset)
        results = []
        for batch in loader:
            batch_out = self.model(**batch, is_eval=True)
            batch["pred"] = batch_out["pred"]
            instances = decompose_batch_into_instances(batch)
            for ins in instances:
                pred_clses = []
                pred_ents = []
                pred_rels = []
                pred_trigger_to_event = defaultdict(
                    lambda: {"event_type": "", "arguments": []}
                )
                pred_events = []
                pred_spans = []
                pred_discon_ents = []
                pred_hyper_rels = []
                raw_schema = ins["raw"]["schema"]
                for multi_part_span in ins["pred"]:
                    span = tuple(multi_part_span)
                    span_to_label = ins["span_to_label"]
                    if span[0] in span_to_label:
                        label = span_to_label[span[0]]
                        if label["task"] == "cls" and len(span) == 1:
                            pred_clses.append(label["string"])
                        elif label["task"] == "ent" and len(span) == 2:
                            string = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_ents.append((label["string"], string))
                        elif label["task"] == "rel" and len(span) == 3:
                            head = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            tail = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_rels.append((label["string"], head, tail))
                        elif label["task"] == "event":
                            if label["type"] == "lm" and len(span) == 2:
                                pred_trigger_to_event[span[1]]["event_type"] = label["string"]  # fmt: skip
                            elif label["type"] == "lr" and len(span) == 3:
                                arg = self._convert_span_to_string(
                                    span[2], ins["input_ids"], self.transform.tokenizer
                                )
                                pred_trigger_to_event[span[1]]["arguments"].append(
                                    {"argument": arg, "role": label["string"]}
                                )
                        elif label["task"] == "discontinuous_ent" and len(span) > 1:
                            parts = [
                                self._convert_span_to_string(
                                    part, ins["input_ids"], self.transform.tokenizer
                                )
                                for part in span[1:]
                            ]
                            string = " ".join([part[0] for part in parts])
                            position = []
                            for part in parts:
                                position.append(part[1])
                            pred_discon_ents.append(
                                (label["string"], string, self.reset_position(position))
                            )
                        elif label["task"] == "hyper_rel" and len(span) == 5 and span[3] in span_to_label:  # fmt: skip
                            q_label = span_to_label[span[3]]
                            span_1 = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            span_2 = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            span_4 = self._convert_span_to_string(
                                span[4], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_hyper_rels.append((label["string"], span_1, span_2, q_label["string"], span_4))  # fmt: skip
                    else:
                        # span task has no labels
                        pred_token_ids = []
                        for part in span:
                            _pred_token_ids = [ins["input_ids"][i] for i in part]
                            pred_token_ids.extend(_pred_token_ids)
                        span_string = self.transform.tokenizer.decode(pred_token_ids)
                        pred_spans.append(
                            (
                                span_string,
                                tuple(
                                    [
                                        tuple(
                                            self.reset_position(
                                                ins["input_ids"].cpu().tolist(), part
                                            )
                                        )
                                        for part in span
                                    ]
                                ),
                            )
                        )
                for trigger, item in pred_trigger_to_event.items():
                    trigger = self._convert_span_to_string(
                        trigger, ins["input_ids"], self.transform.tokenizer
                    )
                    if item["event_type"] not in raw_schema["event"]:
                        continue
                    legal_roles = raw_schema["event"][item["event_type"]]
                    pred_events.append(
                        {
                            "trigger": trigger,
                            "event_type": item["event_type"],
                            "arguments": [
                                arg
                                for arg in filter(
                                    lambda arg: arg["role"] in legal_roles,
                                    item["arguments"],
                                )
                            ],
                        }
                    )
                results.append(
                    {
                        "id": ins["raw"]["id"],
                        "results": {
                            "cls": pred_clses,
                            "ent": pred_ents,
                            "rel": pred_rels,
                            "event": pred_events,
                            "span": pred_spans,
                            "discon_ent": pred_discon_ents,
                            "hyper_rel": pred_hyper_rels,
                        },
                    }
                )

        return results
    
    
    def train(self):
        if self.config.skip_train:
            raise RuntimeError(
                "Training procedure started while config.skip_train is True!"
            )
        else:
            logger.debug("Init optimizer")
            self.optimizer = self.init_optimizer()
            logger.debug(f"optimizer: {self.optimizer}")
            logger.debug("Prepare optimizer")
            self.optimizer = accelerator.prepare_optimizer(self.optimizer)
            logger.debug("Init lr_scheduler")
            self.lr_scheduler = self.init_lr_scheduler()
            if self.lr_scheduler is not None:
                logger.debug(f"lr_scheduler: {type(self.lr_scheduler)}")
                logger.debug("Prepare lr_scheduler")
                self.lr_scheduler = accelerator.prepare_scheduler(self.lr_scheduler)

        if self.config.resumed_training_path is not None:
            self.load(
                self.config.resumed_training_path,
                load_config=False,
                load_model=True,
                load_optimizer=True,
                load_history=True,
            )
            resumed_training = True
        else:
            resumed_training = False
        train_loader = self.get_data_loader("train", False, self.history["curr_epoch"])
        total_steps = self.history["curr_epoch"] * len(train_loader)
        start_time = datetime.now()
        for epoch_idx in range(self.history["curr_epoch"], self.config.num_epochs):
            logger.info(f"Start training {epoch_idx}/{self.config.num_epochs}")
            if not resumed_training:
                self.history["curr_epoch"] = epoch_idx

            used_time = datetime.now() - start_time
            time_per_epoch = safe_division(used_time, self.history["curr_epoch"])
            remain_time = time_per_epoch * (
                self.config.num_epochs - self.history["curr_epoch"]
            )
            logger.info(
                f"Epoch: {epoch_idx}/{self.config.num_epochs} [{str(used_time)}<{str(remain_time)}, {str(time_per_epoch)}/epoch]"
            )

            self.model.train()
            self.optimizer.zero_grad()
            epoch_train_loader = self.get_data_loader(
                "train", is_eval=False, epoch=epoch_idx
            )
            loader = pbar(epoch_train_loader, desc=f"Train(e{epoch_idx})")
            for batch_idx, batch in enumerate(loader):
                with accelerator.accumulate(self.model):
                    if not resumed_training:
                        self.history["curr_batch"] = batch_idx
                        self.history["total_steps"] = total_steps
                    if resumed_training and total_steps < self.history["total_steps"]:
                        total_steps += 1
                        continue
                    elif (
                        resumed_training and total_steps == self.history["total_steps"]
                    ):
                        resumed_training = False

                    result = self.model(**batch)
                    # self.get_error_sample_and_save(batch,result['pred'],self.instruction_path)
                    # self.convert_error_2_Instruction(error_results,'123')
                    # result["loss"] /= self.config.grad_accum_steps
                    accelerator.backward(result["loss"])
                    loss_item = result["loss"].item()
                    self.history["current_train_loss"]["epoch"] += loss_item
                    self.history["current_train_loss"]["step"] += loss_item
                    loader.set_postfix({"loss": loss_item})
                    self.log_loss(
                        self.history["total_steps"], loss_item, "step", "train"
                    )

                    if self.config.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.config.max_grad_norm
                        )
                    # if ((batch_idx + 1) % self.config.grad_accum_steps) == 0 or (
                    #     batch_idx + 1
                    # ) == len(loader):
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    if (
                        self.config.step_eval_interval > 0
                        and (self.history["total_steps"] + 1)
                        % self.config.step_eval_interval
                        == 0
                    ):
                        self._eval_during_train("step")
                        if not self._check_patience():
                            break
                    total_steps += 1
                    if not resumed_training:
                        self.history["total_steps"] += 1

            logger.info(loader)
            if (self.config.epoch_eval_interval > 0) and (
                ((epoch_idx + 1) % self.config.epoch_eval_interval) == 0
            ):
                if epoch_idx > self.config.num_epochs *0.1:
                    self._eval_during_train("epoch")
                    if not self._check_patience():
                        break

        logger.info("Trial finished.")
        if self.config.select_best_by_key == "metric":
            tmp_string = f"{self.history['best_metric']:.5f}"
        else:
            tmp_string = f"{self.history['best_loss']}"
        logger.info(
            f"Best epoch: {self.history['best_epoch']}, step: {self.history['best_step']}"
        )
        logger.info(
            f"Best {self.config.select_best_on_data}.{self.config.select_best_by_key}.{self.config.best_metric_field} : {tmp_string}"
        )

        if self.config.final_eval_on_test:
            logger.info("Loading best ckpt")
            self.load_best_ckpt()
            test_loss, test_measures = self.eval(
                "test", verbose=True, dump=True, postfix="final"
            )
            self.log_loss(0, test_loss, "final", "test")
            self.log_metrics(0, test_measures, "final", "test")
            return test_loss, test_measures

        self.after_whole_train()

        return self.history["best_loss"], self.history["best_metric"]


@register("task")
class SchemaGuidedInstructBertTaskWithPromptReplace(MrcTaggingTask):
    # def __init__(self, config, **kwargs) -> None:
    #     super().__init__(config, **kwargs)

    #     from watchmen import ClientMode, WatchClient

    #     client = WatchClient(
    #         id=config.task_name,
    #         gpus=[4],
    #         req_gpu_num=1,
    #         mode=ClientMode.SCHEDULE,
    #         server_host="127.0.0.1",
    #         server_port=62333,
    #     )
    #     client.wait()

    # def init_lr_scheduler(self):
    #     num_training_steps = int(
    #         631346 / self.config.train_batch_size
    #         * self.config.num_epochs
    #         * accelerator.num_processes
    #     )
    #     num_warmup_steps = math.floor(
    #         num_training_steps * self.config.warmup_proportion
    #     )
    #     # return get_linear_schedule_with_warmup(
    #     return get_cosine_schedule_with_warmup(
    #         self.optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps,
    #     )

    def init_transform(self):
        self.transform: CachedLabelPointerTransformWithPromptReplace
        return CachedLabelPointerTransformWithPromptReplace(
            self.config.max_seq_len,
            self.config.plm_dir,
            mode=self.config.mode,
            label_span=self.config.label_span,
            include_instructions=self.config.get("include_instructions", True),
        )

    def init_data_manager(self):
        if self.config.get("stream_mode", False):
            DatasetClass = StreamReadDatasetWithLen
            transform = self.transform.transform
        else:
            DatasetClass = CachedDataset
            transform = self.transform
        return DataManager(
            self.config.train_filepath,
            self.config.dev_filepath,
            self.config.test_filepath,
            DatasetClass,
            transform,
            load_jsonlines,
            self.config.train_batch_size,
            self.config.eval_batch_size,
            self.transform.collate_fn,
            use_stream_transform=self.config.get("stream_mode", False),
            debug_mode=self.config.debug_mode,
            dump_cache_dir=self.config.dump_cache_dir,
            regenerate_cache=self.config.regenerate_cache,
        )

    def init_model(self):
        self.model = SchemaGuidedInstructBertModelWithPromptReplace_Event(
            self.config.plm_dir,
            vocab_size=len(self.transform.tokenizer),
            use_rope=self.config.use_rope,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
        )
        print(self.config)
        if self.config.get("base_model_path"):
            self.load(
                self.config.base_model_path,
                load_config=False,
                load_model=True,
                load_optimizer=False,
                load_history=False,
            )
        return self.model

    def init_optimizer(self):
        no_decay = r"(embedding|LayerNorm|\.bias$)"
        plm_lr = r"^plm\."
        # non_trainable = r"^plm\.(emb|encoder\.layer\.[0-3])"
        non_trainable = "no_non_trainable"

        param_groups = []
        for name, param in self.model.named_parameters():
            lr = self.config.learning_rate
            weight_decay = self.config.weight_decay
            if re.search(non_trainable, name):
                param.requires_grad = False
            if not re.search(plm_lr, name):
                lr = self.config.other_learning_rate
            if re.search(no_decay, name):
                weight_decay = 0.0
            param_groups.append(
                {"params": param, "lr": lr, "weight_decay": weight_decay}
            )
        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    def init_metric(self):
        return MultiPartSpanMetric()

    def _convert_span_to_string(self, span, token_ids, tokenizer):
        string = ""
        if len(span) == 0 or len(span) > 2:
            pass
        elif len(span) == 1:
            string = tokenizer.decode(token_ids[span[0]])
        elif len(span) == 2:
            string = tokenizer.decode(token_ids[span[0] : span[1] + 1])
        return (string, self.reset_position(token_ids, span))

    def reset_position(self, token_ids: list[int], span: list[int]) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            input_ids = token_ids.cpu().tolist()
        if len(span) < 1:
            return span

        tp_token_id, tl_token_id = self.transform.tokenizer.convert_tokens_to_ids(
            [self.transform.tp_token, self.transform.tl_token]
        )
        offset = 0
        if tp_token_id in input_ids:
            offset = input_ids.index(tp_token_id) + 1
        elif tl_token_id in input_ids:
            offset = input_ids.index(tl_token_id) + 1
        return [i - offset for i in span]

    def predict_api(self, data: list[dict], **kwargs):
        """
        Args:
            data: a list of dict in UDI:
                {
                    "id": str,
                    "instruction": str,
                    "schema": {
                        "ent": list,
                        "rel": list,
                        "event": dict,
                        "cls": list,
                        "discontinuous_ent": list,
                        "hyper_rel": dict
                    },
                    "text": str,
                    "bg": str,
                    "ans": {},  # empty dict
                }
        """
        raw_dataset = [self.transform.transform(d) for d in data]
        loader = self.data_manager.prepare_loader(raw_dataset)
        results = []
        for batch in loader:
            batch_out = self.model(**batch, is_eval=True)
            batch["pred"] = batch_out["pred"]
            instances = decompose_batch_into_instances(batch)
            for ins in instances:
                pred_clses = []
                pred_ents = []
                pred_rels = []
                pred_trigger_to_event = defaultdict(
                    lambda: {"event_type": "", "arguments": []}
                )
                pred_events = []
                pred_spans = []
                pred_discon_ents = []
                pred_hyper_rels = []
                raw_schema = ins["raw"]["schema"]
                for multi_part_span in ins["pred"]:
                    span = tuple(multi_part_span)
                    span_to_label = ins["span_to_label"]
                    if span[0] in span_to_label:
                        label = span_to_label[span[0]]
                        if label["task"] == "cls" and len(span) == 1:
                            pred_clses.append(label["string"])
                        elif label["task"] == "ent" and len(span) == 2:
                            string = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_ents.append((label["string"], string))
                        elif label["task"] == "rel" and len(span) == 3:
                            head = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            tail = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_rels.append((label["string"], head, tail))
                        elif label["task"] == "event":
                            if label["type"] == "lm" and len(span) == 2:
                                pred_trigger_to_event[span[1]]["event_type"] = label["string"]  # fmt: skip
                            elif label["type"] == "lr" and len(span) == 3:
                                arg = self._convert_span_to_string(
                                    span[2], ins["input_ids"], self.transform.tokenizer
                                )
                                pred_trigger_to_event[span[1]]["arguments"].append(
                                    {"argument": arg, "role": label["string"]}
                                )
                        elif label["task"] == "discontinuous_ent" and len(span) > 1:
                            parts = [
                                self._convert_span_to_string(
                                    part, ins["input_ids"], self.transform.tokenizer
                                )
                                for part in span[1:]
                            ]
                            string = " ".join([part[0] for part in parts])
                            position = []
                            for part in parts:
                                position.append(part[1])
                            pred_discon_ents.append(
                                (label["string"], string, self.reset_position(position))
                            )
                        elif label["task"] == "hyper_rel" and len(span) == 5 and span[3] in span_to_label:  # fmt: skip
                            q_label = span_to_label[span[3]]
                            span_1 = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            span_2 = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            span_4 = self._convert_span_to_string(
                                span[4], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_hyper_rels.append((label["string"], span_1, span_2, q_label["string"], span_4))  # fmt: skip
                    else:
                        # span task has no labels
                        pred_token_ids = []
                        for part in span:
                            _pred_token_ids = [ins["input_ids"][i] for i in part]
                            pred_token_ids.extend(_pred_token_ids)
                        span_string = self.transform.tokenizer.decode(pred_token_ids)
                        pred_spans.append(
                            (
                                span_string,
                                tuple(
                                    [
                                        tuple(
                                            self.reset_position(
                                                ins["input_ids"].cpu().tolist(), part
                                            )
                                        )
                                        for part in span
                                    ]
                                ),
                            )
                        )
                for trigger, item in pred_trigger_to_event.items():
                    trigger = self._convert_span_to_string(
                        trigger, ins["input_ids"], self.transform.tokenizer
                    )
                    if item["event_type"] not in raw_schema["event"]:
                        continue
                    legal_roles = raw_schema["event"][item["event_type"]]
                    pred_events.append(
                        {
                            "trigger": trigger,
                            "event_type": item["event_type"],
                            "arguments": [
                                arg
                                for arg in filter(
                                    lambda arg: arg["role"] in legal_roles,
                                    item["arguments"],
                                )
                            ],
                        }
                    )
                results.append(
                    {
                        "id": ins["raw"]["id"],
                        "results": {
                            "cls": pred_clses,
                            "ent": pred_ents,
                            "rel": pred_rels,
                            "event": pred_events,
                            "span": pred_spans,
                            "discon_ent": pred_discon_ents,
                            "hyper_rel": pred_hyper_rels,
                        },
                    }
                )

        return results
    def train(self):
        if self.config.skip_train:
            raise RuntimeError(
                "Training procedure started while config.skip_train is True!"
            )
        else:
            logger.debug("Init optimizer")
            self.optimizer = self.init_optimizer()
            logger.debug(f"optimizer: {self.optimizer}")
            logger.debug("Prepare optimizer")
            self.optimizer = accelerator.prepare_optimizer(self.optimizer)
            logger.debug("Init lr_scheduler")
            self.lr_scheduler = self.init_lr_scheduler()
            if self.lr_scheduler is not None:
                logger.debug(f"lr_scheduler: {type(self.lr_scheduler)}")
                logger.debug("Prepare lr_scheduler")
                self.lr_scheduler = accelerator.prepare_scheduler(self.lr_scheduler)

        if self.config.resumed_training_path is not None:
            self.load(
                self.config.resumed_training_path,
                load_config=False,
                load_model=True,
                load_optimizer=True,
                load_history=True,
            )
            resumed_training = True
        else:
            resumed_training = False
        train_loader = self.get_data_loader("train", False, self.history["curr_epoch"])
        total_steps = self.history["curr_epoch"] * len(train_loader)
        start_time = datetime.now()
        hidden_prmopt = None
        for epoch_idx in range(self.history["curr_epoch"], self.config.num_epochs):

            logger.info(f"Start training {epoch_idx}/{self.config.num_epochs}")
            if not resumed_training:
                self.history["curr_epoch"] = epoch_idx

            used_time = datetime.now() - start_time
            time_per_epoch = safe_division(used_time, self.history["curr_epoch"])
            remain_time = time_per_epoch * (
                self.config.num_epochs - self.history["curr_epoch"]
            )
            logger.info(
                f"Epoch: {epoch_idx}/{self.config.num_epochs} [{str(used_time)}<{str(remain_time)}, {str(time_per_epoch)}/epoch]"
            )

            self.model.train()
            self.optimizer.zero_grad()
            epoch_train_loader = self.get_data_loader(
                "train", is_eval=False, epoch=epoch_idx
            )
            loader = pbar(epoch_train_loader, desc=f"Train(e{epoch_idx})")
            for batch_idx, batch in enumerate(loader):
                with accelerator.accumulate(self.model):
                    if not resumed_training:
                        self.history["curr_batch"] = batch_idx
                        self.history["total_steps"] = total_steps
                    if resumed_training and total_steps < self.history["total_steps"]:
                        total_steps += 1
                        continue
                    elif (
                        resumed_training and total_steps == self.history["total_steps"]
                    ):
                        resumed_training = False
                    #1.get the input from batch
                    #2.return the hidden_[Mask] vector respectively
                    #3.inject to the input
                    
                    with torch.no_grad():
                        if hidden_prmopt is None:
                            prompts_ids = batch['prompts_ids']
                            prompts_mask = batch['prompts_mask']
                            hidden_prmopt = self.model.get_prompt_hidden(prompts_ids[0],prompts_mask[0])
                        batch['hidden_prmopt'] = hidden_prmopt.detach()
                        # torch.cuda.empty_cache()
                    
                    
                    result = self.model(**batch)
                    # self.get_error_sample_and_save(batch,result['pred'],self.instruction_path)
                    # self.convert_error_2_Instruction(error_results,'123')
                    # result["loss"] /= self.config.grad_accum_steps
                    accelerator.backward(result["loss"])
                    loss_item = result["loss"].item()
                    self.history["current_train_loss"]["epoch"] += loss_item
                    self.history["current_train_loss"]["step"] += loss_item
                    loader.set_postfix({"loss": loss_item})
                    self.log_loss(
                        self.history["total_steps"], loss_item, "step", "train"
                    )

                    if self.config.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.config.max_grad_norm
                        )
                    # if ((batch_idx + 1) % self.config.grad_accum_steps) == 0 or (
                    #     batch_idx + 1
                    # ) == len(loader):
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    if (
                        self.config.step_eval_interval > 0
                        and (self.history["total_steps"] + 1)
                        % self.config.step_eval_interval
                        == 0 and epoch_idx>2
                    ):
                        self._eval_during_train("step")
                        if not self._check_patience():
                            break
                    total_steps += 1
                    if not resumed_training:
                        self.history["total_steps"] += 1

            logger.info(loader)
            if (self.config.epoch_eval_interval > 0) and (
                ((epoch_idx + 1) % self.config.epoch_eval_interval) == 0 and epoch_idx>2
            ):
                if epoch_idx > self.config.num_epochs *0.1:
                    self._eval_during_train("epoch")
                    if not self._check_patience():
                        break

        logger.info("Trial finished.")
        if self.config.select_best_by_key == "metric":
            tmp_string = f"{self.history['best_metric']:.5f}"
        else:
            tmp_string = f"{self.history['best_loss']}"
        logger.info(
            f"Best epoch: {self.history['best_epoch']}, step: {self.history['best_step']}"
        )
        logger.info(
            f"Best {self.config.select_best_on_data}.{self.config.select_best_by_key}.{self.config.best_metric_field} : {tmp_string}"
        )

        if self.config.final_eval_on_test:
            logger.info("Loading best ckpt")
            self.load_best_ckpt()
            test_loss, test_measures = self.eval(
                "test", verbose=True, dump=True, postfix="final"
            )
            self.log_loss(0, test_loss, "final", "test")
            self.log_metrics(0, test_measures, "final", "test")
            return test_loss, test_measures

        self.after_whole_train()

        return self.history["best_loss"], self.history["best_metric"]
    from typing import Tuple
    @torch.no_grad()
    def eval(
        self, dataset_name, verbose=False, dump=False, dump_middle=False, postfix=""
    ) -> Tuple[float, dict]:
        from rex.utils.dict import get_dict_content
        from rex.utils.io import dump_json, dump_jsonlines
        """Eval on specific dataset and return loss and measurements

        Args:
            dataset_name: which dataset to evaluate
            verbose: whether to log evaluation results
            dump: if True, dump metric results to `self.measures_path`
            dump_middle: if True, dump middle results to `self.middle_path`
            postfix: filepath postfix for dumping

        Returns:
            eval_loss: float
            metrics: dict
        """
        self.model.eval()
        eval_loader = self.get_data_loader(
            dataset_name, is_eval=True, epoch=self.history["curr_epoch"]
        )
        loader = pbar(eval_loader, desc=f"{dataset_name} - {postfix} Eval", ascii=True)

        eval_loss = 0.0
        tot_batch_results = []
        hidden_prmopt = None
        for batch in loader:
            with torch.no_grad():
                if hidden_prmopt is None:
                    prompts_ids = batch['prompts_ids']
                    prompts_mask = batch['prompts_mask']
                    hidden_prmopt = self.model.get_prompt_hidden(prompts_ids[0],prompts_mask[0])
                    torch.cuda.empty_cache()
            batch['hidden_prmopt'] = hidden_prmopt.detach()

            out = self.model(**batch, is_eval=True)

            eval_loss += out["loss"].item()
            if 'hidden_prmopt' in batch:
                del batch['hidden_prmopt']
            batch_results: dict = self.metric(batch, out)
            batch_metric_score = get_dict_content(
                batch_results["metric_scores"], self.config.best_metric_field
            )
            loader.set_postfix({self.config.best_metric_field: batch_metric_score})

            batch_instances = [
                {"gold": gold, "pred": pred}
                for gold, pred in zip(batch_results["gold"], batch_results["pred"])
            ]
            tot_batch_results.extend(batch_instances)

        logger.info(loader)
        measurements = self.metric.compute()

        if verbose:
            logger.info(f"Eval dataset: {dataset_name}")
            logger.info(f"Eval loss: {eval_loss}")
            logger.info(
                f"Eval metrics: {get_dict_content(measurements, self.config.best_metric_field)}"
            )
        _filename_prefix = (
            f"{dataset_name}.{postfix}" if len(postfix) > 0 else f"{dataset_name}"
        )
        if dump:
            dump_obj = {
                "dataset_name": dataset_name,
                "eval_loss": eval_loss,
                "metrics": measurements,
            }
            _measure_result_filepath = self.measures_path.joinpath(
                f"{_filename_prefix}.json"
            )
            dump_json(dump_obj, _measure_result_filepath)
            logger.info(f"Dump measure results into {_measure_result_filepath}")
        if dump_middle:
            _middle_result_filepath = self.middle_path.joinpath(
                f"{_filename_prefix}.jsonl"
            )
            dump_jsonlines(tot_batch_results, _middle_result_filepath)
            logger.info(f"Dump middle results into {_middle_result_filepath}")

        self.metric.reset()

        return eval_loss, measurements


@register("task")
class SchemaGuidedInstructBertTaskWith_PN_PromptReplace(MrcTaggingTask):
    # def __init__(self, config, **kwargs) -> None:
    #     super().__init__(config, **kwargs)

    #     from watchmen import ClientMode, WatchClient

    #     client = WatchClient(
    #         id=config.task_name,
    #         gpus=[4],
    #         req_gpu_num=1,
    #         mode=ClientMode.SCHEDULE,
    #         server_host="127.0.0.1",
    #         server_port=62333,
    #     )
    #     client.wait()

    # def init_lr_scheduler(self):
    #     num_training_steps = int(
    #         631346 / self.config.train_batch_size
    #         * self.config.num_epochs
    #         * accelerator.num_processes
    #     )
    #     num_warmup_steps = math.floor(
    #         num_training_steps * self.config.warmup_proportion
    #     )
    #     # return get_linear_schedule_with_warmup(
    #     return get_cosine_schedule_with_warmup(
    #         self.optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps,
    #     )

    def init_transform(self):
        self.transform: CachedLabelPointerTransformWith_PN_PromptReplace
        return CachedLabelPointerTransformWith_PN_PromptReplace(
            self.config.max_seq_len,
            self.config.plm_dir,
            mode=self.config.mode,
            label_span=self.config.label_span,
            include_instructions=self.config.get("include_instructions", True),
        )

    def init_data_manager(self):
        if self.config.get("stream_mode", False):
            DatasetClass = StreamReadDatasetWithLen
            transform = self.transform.transform
        else:
            DatasetClass = CachedDataset
            transform = self.transform
        return DataManager(
            self.config.train_filepath,
            self.config.dev_filepath,
            self.config.test_filepath,
            DatasetClass,
            transform,
            load_jsonlines,
            self.config.train_batch_size,
            self.config.eval_batch_size,
            self.transform.collate_fn,
            use_stream_transform=self.config.get("stream_mode", False),
            debug_mode=self.config.debug_mode,
            dump_cache_dir=self.config.dump_cache_dir,
            regenerate_cache=self.config.regenerate_cache,
        )

    def init_model(self):
        self.model = SchemaGuidedInstructBertModelWithPromptReplace_PN_Event(
            self.config.plm_dir,
            vocab_size=len(self.transform.tokenizer),
            use_rope=self.config.use_rope,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
        )
        print(self.config)
        if self.config.get("base_model_path"):
            self.load(
                self.config.base_model_path,
                load_config=False,
                load_model=True,
                load_optimizer=False,
                load_history=False,
            )
        return self.model

    def init_optimizer(self):
        no_decay = r"(embedding|LayerNorm|\.bias$)"
        plm_lr = r"^plm\."
        # non_trainable = r"^plm\.(emb|encoder\.layer\.[0-3])"
        non_trainable = "no_non_trainable"

        param_groups = []
        for name, param in self.model.named_parameters():
            lr = self.config.learning_rate
            weight_decay = self.config.weight_decay
            if re.search(non_trainable, name):
                param.requires_grad = False
            if not re.search(plm_lr, name):
                lr = self.config.other_learning_rate
            if re.search(no_decay, name):
                weight_decay = 0.0
            param_groups.append(
                {"params": param, "lr": lr, "weight_decay": weight_decay}
            )
        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    def init_metric(self):
        return MultiPartSpanMetric()

    def _convert_span_to_string(self, span, token_ids, tokenizer):
        string = ""
        if len(span) == 0 or len(span) > 2:
            pass
        elif len(span) == 1:
            string = tokenizer.decode(token_ids[span[0]])
        elif len(span) == 2:
            string = tokenizer.decode(token_ids[span[0] : span[1] + 1])
        return (string, self.reset_position(token_ids, span))

    def reset_position(self, token_ids: list[int], span: list[int]) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            input_ids = token_ids.cpu().tolist()
        if len(span) < 1:
            return span

        tp_token_id, tl_token_id = self.transform.tokenizer.convert_tokens_to_ids(
            [self.transform.tp_token, self.transform.tl_token]
        )
        offset = 0
        if tp_token_id in input_ids:
            offset = input_ids.index(tp_token_id) + 1
        elif tl_token_id in input_ids:
            offset = input_ids.index(tl_token_id) + 1
        return [i - offset for i in span]

    def predict_api(self, data: list[dict], **kwargs):
        """
        Args:
            data: a list of dict in UDI:
                {
                    "id": str,
                    "instruction": str,
                    "schema": {
                        "ent": list,
                        "rel": list,
                        "event": dict,
                        "cls": list,
                        "discontinuous_ent": list,
                        "hyper_rel": dict
                    },
                    "text": str,
                    "bg": str,
                    "ans": {},  # empty dict
                }
        """
        raw_dataset = [self.transform.transform(d) for d in data]
        loader = self.data_manager.prepare_loader(raw_dataset)
        results = []
        for batch in loader:
            batch_out = self.model(**batch, is_eval=True)
            batch["pred"] = batch_out["pred"]
            instances = decompose_batch_into_instances(batch)
            for ins in instances:
                pred_clses = []
                pred_ents = []
                pred_rels = []
                pred_trigger_to_event = defaultdict(
                    lambda: {"event_type": "", "arguments": []}
                )
                pred_events = []
                pred_spans = []
                pred_discon_ents = []
                pred_hyper_rels = []
                raw_schema = ins["raw"]["schema"]
                for multi_part_span in ins["pred"]:
                    span = tuple(multi_part_span)
                    span_to_label = ins["span_to_label"]
                    if span[0] in span_to_label:
                        label = span_to_label[span[0]]
                        if label["task"] == "cls" and len(span) == 1:
                            pred_clses.append(label["string"])
                        elif label["task"] == "ent" and len(span) == 2:
                            string = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_ents.append((label["string"], string))
                        elif label["task"] == "rel" and len(span) == 3:
                            head = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            tail = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_rels.append((label["string"], head, tail))
                        elif label["task"] == "event":
                            if label["type"] == "lm" and len(span) == 2:
                                pred_trigger_to_event[span[1]]["event_type"] = label["string"]  # fmt: skip
                            elif label["type"] == "lr" and len(span) == 3:
                                arg = self._convert_span_to_string(
                                    span[2], ins["input_ids"], self.transform.tokenizer
                                )
                                pred_trigger_to_event[span[1]]["arguments"].append(
                                    {"argument": arg, "role": label["string"]}
                                )
                        elif label["task"] == "discontinuous_ent" and len(span) > 1:
                            parts = [
                                self._convert_span_to_string(
                                    part, ins["input_ids"], self.transform.tokenizer
                                )
                                for part in span[1:]
                            ]
                            string = " ".join([part[0] for part in parts])
                            position = []
                            for part in parts:
                                position.append(part[1])
                            pred_discon_ents.append(
                                (label["string"], string, self.reset_position(position))
                            )
                        elif label["task"] == "hyper_rel" and len(span) == 5 and span[3] in span_to_label:  # fmt: skip
                            q_label = span_to_label[span[3]]
                            span_1 = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            span_2 = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            span_4 = self._convert_span_to_string(
                                span[4], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_hyper_rels.append((label["string"], span_1, span_2, q_label["string"], span_4))  # fmt: skip
                    else:
                        # span task has no labels
                        pred_token_ids = []
                        for part in span:
                            _pred_token_ids = [ins["input_ids"][i] for i in part]
                            pred_token_ids.extend(_pred_token_ids)
                        span_string = self.transform.tokenizer.decode(pred_token_ids)
                        pred_spans.append(
                            (
                                span_string,
                                tuple(
                                    [
                                        tuple(
                                            self.reset_position(
                                                ins["input_ids"].cpu().tolist(), part
                                            )
                                        )
                                        for part in span
                                    ]
                                ),
                            )
                        )
                for trigger, item in pred_trigger_to_event.items():
                    trigger = self._convert_span_to_string(
                        trigger, ins["input_ids"], self.transform.tokenizer
                    )
                    if item["event_type"] not in raw_schema["event"]:
                        continue
                    legal_roles = raw_schema["event"][item["event_type"]]
                    pred_events.append(
                        {
                            "trigger": trigger,
                            "event_type": item["event_type"],
                            "arguments": [
                                arg
                                for arg in filter(
                                    lambda arg: arg["role"] in legal_roles,
                                    item["arguments"],
                                )
                            ],
                        }
                    )
                results.append(
                    {
                        "id": ins["raw"]["id"],
                        "results": {
                            "cls": pred_clses,
                            "ent": pred_ents,
                            "rel": pred_rels,
                            "event": pred_events,
                            "span": pred_spans,
                            "discon_ent": pred_discon_ents,
                            "hyper_rel": pred_hyper_rels,
                        },
                    }
                )

        return results
    def train(self):
        if self.config.skip_train:
            raise RuntimeError(
                "Training procedure started while config.skip_train is True!"
            )
        else:
            logger.debug("Init optimizer")
            self.optimizer = self.init_optimizer()
            logger.debug(f"optimizer: {self.optimizer}")
            logger.debug("Prepare optimizer")
            self.optimizer = accelerator.prepare_optimizer(self.optimizer)
            logger.debug("Init lr_scheduler")
            self.lr_scheduler = self.init_lr_scheduler()
            if self.lr_scheduler is not None:
                logger.debug(f"lr_scheduler: {type(self.lr_scheduler)}")
                logger.debug("Prepare lr_scheduler")
                self.lr_scheduler = accelerator.prepare_scheduler(self.lr_scheduler)

        if self.config.resumed_training_path is not None:
            self.load(
                self.config.resumed_training_path,
                load_config=False,
                load_model=True,
                load_optimizer=True,
                load_history=True,
            )
            resumed_training = True
        else:
            resumed_training = False
        train_loader = self.get_data_loader("train", False, self.history["curr_epoch"])
        total_steps = self.history["curr_epoch"] * len(train_loader)
        start_time = datetime.now()
        for epoch_idx in range(self.history["curr_epoch"], self.config.num_epochs):
            hidden_prmopt = None
            hidden_prmopt_neg = None
            logger.info(f"Start training {epoch_idx}/{self.config.num_epochs}")
            if not resumed_training:
                self.history["curr_epoch"] = epoch_idx

            used_time = datetime.now() - start_time
            time_per_epoch = safe_division(used_time, self.history["curr_epoch"])
            remain_time = time_per_epoch * (
                self.config.num_epochs - self.history["curr_epoch"]
            )
            logger.info(
                f"Epoch: {epoch_idx}/{self.config.num_epochs} [{str(used_time)}<{str(remain_time)}, {str(time_per_epoch)}/epoch]"
            )

            self.model.train()
            self.optimizer.zero_grad()
            epoch_train_loader = self.get_data_loader(
                "train", is_eval=False, epoch=epoch_idx
            )
            loader = pbar(epoch_train_loader, desc=f"Train(e{epoch_idx})")
            for batch_idx, batch in enumerate(loader):
                with accelerator.accumulate(self.model):
                    if not resumed_training:
                        self.history["curr_batch"] = batch_idx
                        self.history["total_steps"] = total_steps
                    if resumed_training and total_steps < self.history["total_steps"]:
                        total_steps += 1
                        continue
                    elif (
                        resumed_training and total_steps == self.history["total_steps"]
                    ):
                        resumed_training = False
                    #1.get the input from batch
                    #2.return the hidden_[Mask] vector respectively
                    #3.inject to the input
                    
                    with torch.no_grad():
                        if hidden_prmopt is None:
                            prompts_ids = batch['prompts_ids']
                            prompts_mask = batch['prompts_mask']
                            hidden_prmopt = self.model.get_prompt_hidden(prompts_ids[0],prompts_mask[0])
                            
                            prompts_ids_neg = batch['prompts_ids_neg']
                            prompts_mask_neg = batch['prompts_mask_neg']
                            hidden_prmopt_neg = self.model.get_prompt_hidden(prompts_ids_neg[0],prompts_mask_neg[0])
            
                        batch['hidden_prmopt'] = hidden_prmopt.detach()
                        batch['hidden_prmopt_neg'] = hidden_prmopt_neg.detach()
                        
                    
                    result = self.model(**batch)
                    # self.get_error_sample_and_save(batch,result['pred'],self.instruction_path)
                    # self.convert_error_2_Instruction(error_results,'123')
                    # result["loss"] /= self.config.grad_accum_steps
                    accelerator.backward(result["loss"])
                    loss_item = result["loss"].item()
                    self.history["current_train_loss"]["epoch"] += loss_item
                    self.history["current_train_loss"]["step"] += loss_item
                    loader.set_postfix({"loss": loss_item})
                    self.log_loss(
                        self.history["total_steps"], loss_item, "step", "train"
                    )

                    if self.config.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.config.max_grad_norm
                        )
                    # if ((batch_idx + 1) % self.config.grad_accum_steps) == 0 or (
                    #     batch_idx + 1
                    # ) == len(loader):
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    if (
                        self.config.step_eval_interval > 0
                        and (self.history["total_steps"] + 1)
                        % self.config.step_eval_interval
                        == 0 and epoch_idx > 2
                    ):
                        self._eval_during_train("step")
                        if not self._check_patience():
                            break
                    total_steps += 1
                    if not resumed_training:
                        self.history["total_steps"] += 1

            logger.info(loader)
            if (self.config.epoch_eval_interval > 0) and (
                ((epoch_idx + 1) % self.config.epoch_eval_interval) == 0 and epoch_idx>2
            ):
                if epoch_idx > self.config.num_epochs*0.1:
                    self._eval_during_train("epoch")
                    if not self._check_patience():
                        break

        logger.info("Trial finished.")
        if self.config.select_best_by_key == "metric":
            tmp_string = f"{self.history['best_metric']:.5f}"
        else:
            tmp_string = f"{self.history['best_loss']}"
        logger.info(
            f"Best epoch: {self.history['best_epoch']}, step: {self.history['best_step']}"
        )
        logger.info(
            f"Best {self.config.select_best_on_data}.{self.config.select_best_by_key}.{self.config.best_metric_field} : {tmp_string}"
        )

        if self.config.final_eval_on_test:
            logger.info("Loading best ckpt")
            self.load_best_ckpt()
            test_loss, test_measures = self.eval(
                "test", verbose=True, dump=True, postfix="final"
            )
            self.log_loss(0, test_loss, "final", "test")
            self.log_metrics(0, test_measures, "final", "test")
            return test_loss, test_measures

        self.after_whole_train()

        return self.history["best_loss"], self.history["best_metric"]
    from typing import Tuple
    @torch.no_grad()
    def eval(
        self, dataset_name, verbose=False, dump=False, dump_middle=False, postfix=""
    ) -> Tuple[float, dict]:
        from rex.utils.dict import get_dict_content
        from rex.utils.io import dump_json, dump_jsonlines
        """Eval on specific dataset and return loss and measurements

        Args:
            dataset_name: which dataset to evaluate
            verbose: whether to log evaluation results
            dump: if True, dump metric results to `self.measures_path`
            dump_middle: if True, dump middle results to `self.middle_path`
            postfix: filepath postfix for dumping

        Returns:
            eval_loss: float
            metrics: dict
        """
        self.model.eval()
        eval_loader = self.get_data_loader(
            dataset_name, is_eval=True, epoch=self.history["curr_epoch"]
        )
        loader = pbar(eval_loader, desc=f"{dataset_name} - {postfix} Eval", ascii=True)

        eval_loss = 0.0
        tot_batch_results = []
        hidden_prmopt = None
        hidden_prmopt_neg = None
        for batch in loader:
            with torch.no_grad():
                if hidden_prmopt is None:
                    prompts_ids = batch['prompts_ids']
                    prompts_mask = batch['prompts_mask']
                    hidden_prmopt = self.model.get_prompt_hidden(prompts_ids[0],prompts_mask[0])
                    prompts_ids_neg = batch['prompts_ids_neg']
                    prompts_mask_neg = batch['prompts_mask_neg']
                    hidden_prmopt_neg = self.model.get_prompt_hidden(prompts_ids_neg[0],prompts_mask_neg[0])
                
                batch['hidden_prmopt'] = hidden_prmopt.detach()
                batch['hidden_prmopt_neg'] = hidden_prmopt_neg.detach()

            out = self.model(**batch, is_eval=True)

            eval_loss += out["loss"].item()
            if 'hidden_prmopt' in batch:
                del batch['hidden_prmopt']
                del batch['hidden_prmopt_neg']
            batch_results: dict = self.metric(batch, out)
            batch_metric_score = get_dict_content(
                batch_results["metric_scores"], self.config.best_metric_field
            )
            loader.set_postfix({self.config.best_metric_field: batch_metric_score})

            batch_instances = [
                {"gold": gold, "pred": pred}
                for gold, pred in zip(batch_results["gold"], batch_results["pred"])
            ]
            tot_batch_results.extend(batch_instances)

        logger.info(loader)
        measurements = self.metric.compute()

        if verbose:
            logger.info(f"Eval dataset: {dataset_name}")
            logger.info(f"Eval loss: {eval_loss}")
            logger.info(
                f"Eval metrics: {get_dict_content(measurements, self.config.best_metric_field)}"
            )
        _filename_prefix = (
            f"{dataset_name}.{postfix}" if len(postfix) > 0 else f"{dataset_name}"
        )
        if dump:
            dump_obj = {
                "dataset_name": dataset_name,
                "eval_loss": eval_loss,
                "metrics": measurements,
            }
            _measure_result_filepath = self.measures_path.joinpath(
                f"{_filename_prefix}.json"
            )
            dump_json(dump_obj, _measure_result_filepath)
            logger.info(f"Dump measure results into {_measure_result_filepath}")
        if dump_middle:
            _middle_result_filepath = self.middle_path.joinpath(
                f"{_filename_prefix}.jsonl"
            )
            dump_jsonlines(tot_batch_results, _middle_result_filepath)
            logger.info(f"Dump middle results into {_middle_result_filepath}")

        self.metric.reset()

        return eval_loss, measurements

@register("task")
class SchemaGuidedInstructBertTaskWithPromptReplace_Layer_Level(MrcTaggingTask):
    # def __init__(self, config, **kwargs) -> None:
    #     super().__init__(config, **kwargs)

    #     from watchmen import ClientMode, WatchClient

    #     client = WatchClient(
    #         id=config.task_name,
    #         gpus=[4],
    #         req_gpu_num=1,
    #         mode=ClientMode.SCHEDULE,
    #         server_host="127.0.0.1",
    #         server_port=62333,
    #     )
    #     client.wait()

    # def init_lr_scheduler(self):
    #     num_training_steps = int(
    #         631346 / self.config.train_batch_size
    #         * self.config.num_epochs
    #         * accelerator.num_processes
    #     )
    #     num_warmup_steps = math.floor(
    #         num_training_steps * self.config.warmup_proportion
    #     )
    #     # return get_linear_schedule_with_warmup(
    #     return get_cosine_schedule_with_warmup(
    #         self.optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps,
    #     )

    def init_transform(self):
        self.transform: CachedLabelPointerTransformWithPromptReplace
        return CachedLabelPointerTransformWithPromptReplace(
            self.config.max_seq_len,
            self.config.plm_dir,
            mode=self.config.mode,
            label_span=self.config.label_span,
            include_instructions=self.config.get("include_instructions", True),
        )

    def init_data_manager(self):
        if self.config.get("stream_mode", False):
            DatasetClass = StreamReadDatasetWithLen
            transform = self.transform.transform
        else:
            DatasetClass = CachedDataset
            transform = self.transform
        return DataManager(
            self.config.train_filepath,
            self.config.dev_filepath,
            self.config.test_filepath,
            DatasetClass,
            transform,
            load_jsonlines,
            self.config.train_batch_size,
            self.config.eval_batch_size,
            self.transform.collate_fn,
            use_stream_transform=self.config.get("stream_mode", False),
            debug_mode=self.config.debug_mode,
            dump_cache_dir=self.config.dump_cache_dir,
            regenerate_cache=self.config.regenerate_cache,
        )

    def init_model(self):
        self.model = SchemaGuidedInstructBertModelWithPromptReplace_Event_Layer_Level(
            self.config.plm_dir,
            vocab_size=len(self.transform.tokenizer),
            use_rope=self.config.use_rope,
            biaffine_size=self.config.biaffine_size,
            dropout=self.config.dropout,
        )
        # print(self.config)
        if self.config.get("base_model_path"):
            self.load(
                self.config.base_model_path,
                load_config=False,
                load_model=True,
                load_optimizer=False,
                load_history=False,
            )
        return self.model

    def load(
        self,
        path: str,
        load_config: Optional[bool] = False,
        load_model: Optional[bool] = True,
        load_optimizer: Optional[bool] = False,
        load_history: Optional[bool] = True,
    ):
        if Path(path).exists():
            logger.info("Resume checkpoint from {}".format(path))
        else:
            raise ValueError("Checkpoint does not exist, {}".format(path))

        if torch.cuda.device_count() == 0:
            logger.debug("Load store_dict into cpu")
            store_dict = torch.load(path, map_location="cpu")
        else:
            logger.debug(f"Load store_dict into {accelerator.device}")
            store_dict = torch.load(path, map_location=accelerator.device)

        if load_config:
            self.config = OmegaConf.load(self.task_path / CONFIG_PARAMS_FILENAME)

        if load_model:
            if self.model and "model_state" in store_dict:
                unwrapped_model = accelerator.unwrap_model(self.model)
                # unwrapped_model.load_state_dict(store_dict["model_state"],strict=False)
                unwrapped_model = from_local_pretrained(unwrapped_model,store_dict["model_state"])
                logger.debug("Load model successfully")
            else:
                raise ValueError(
                    f"Model loading failed. self.model={self.model}, stored_dict_keys={store_dict.keys()}"
                )
        else:
            logger.debug("Not load model")

        if load_optimizer:
            if self.optimizer and "optimizer_state" in store_dict:
                self.optimizer.load_state_dict(store_dict["optimizer_state"])
                logger.debug("Load optimizer successfully")
            else:
                raise ValueError(
                    f"Model loading failed. self.optimizer={self.optimizer}, stored_dict_keys={store_dict.keys()}"
                )
        else:
            logger.debug("Not load optimizer")

        if load_history:
            history = store_dict.pop("history")
            self.reset_history(reset_all=True)
            if history is not None:
                self.history = history
            else:
                logger.debug(
                    "Loaded history is None, reset history to empty.", level="WARNING"
                )

    def init_optimizer(self):
        no_decay = r"(embedding|LayerNorm|\.bias$)"
        plm_lr = r"^plm\."
        # non_trainable = r"^plm\.(emb|encoder\.layer\.[0-3])"
        non_trainable = "no_non_trainable"

        param_groups = []
        for name, param in self.model.named_parameters():
            lr = self.config.learning_rate
            weight_decay = self.config.weight_decay
            if re.search(non_trainable, name):
                param.requires_grad = False
            if not re.search(plm_lr, name):
                lr = self.config.other_learning_rate
            if re.search(no_decay, name):
                weight_decay = 0.0
            param_groups.append(
                {"params": param, "lr": lr, "weight_decay": weight_decay}
            )
        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    def init_metric(self):
        return MultiPartSpanMetric()

    def _convert_span_to_string(self, span, token_ids, tokenizer):
        string = ""
        if len(span) == 0 or len(span) > 2:
            pass
        elif len(span) == 1:
            string = tokenizer.decode(token_ids[span[0]])
        elif len(span) == 2:
            string = tokenizer.decode(token_ids[span[0] : span[1] + 1])
        return (string, self.reset_position(token_ids, span))

    def reset_position(self, token_ids: list[int], span: list[int]) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            input_ids = token_ids.cpu().tolist()
        if len(span) < 1:
            return span

        tp_token_id, tl_token_id = self.transform.tokenizer.convert_tokens_to_ids(
            [self.transform.tp_token, self.transform.tl_token]
        )
        offset = 0
        if tp_token_id in input_ids:
            offset = input_ids.index(tp_token_id) + 1
        elif tl_token_id in input_ids:
            offset = input_ids.index(tl_token_id) + 1
        return [i - offset for i in span]

    def predict_api(self, data: list[dict], **kwargs):
        """
        Args:
            data: a list of dict in UDI:
                {
                    "id": str,
                    "instruction": str,
                    "schema": {
                        "ent": list,
                        "rel": list,
                        "event": dict,
                        "cls": list,
                        "discontinuous_ent": list,
                        "hyper_rel": dict
                    },
                    "text": str,
                    "bg": str,
                    "ans": {},  # empty dict
                }
        """
        raw_dataset = [self.transform.transform(d) for d in data]
        loader = self.data_manager.prepare_loader(raw_dataset)
        results = []
        for batch in loader:
            batch_out = self.model(**batch, is_eval=True)
            batch["pred"] = batch_out["pred"]
            instances = decompose_batch_into_instances(batch)
            for ins in instances:
                pred_clses = []
                pred_ents = []
                pred_rels = []
                pred_trigger_to_event = defaultdict(
                    lambda: {"event_type": "", "arguments": []}
                )
                pred_events = []
                pred_spans = []
                pred_discon_ents = []
                pred_hyper_rels = []
                raw_schema = ins["raw"]["schema"]
                for multi_part_span in ins["pred"]:
                    span = tuple(multi_part_span)
                    span_to_label = ins["span_to_label"]
                    if span[0] in span_to_label:
                        label = span_to_label[span[0]]
                        if label["task"] == "cls" and len(span) == 1:
                            pred_clses.append(label["string"])
                        elif label["task"] == "ent" and len(span) == 2:
                            string = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_ents.append((label["string"], string))
                        elif label["task"] == "rel" and len(span) == 3:
                            head = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            tail = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_rels.append((label["string"], head, tail))
                        elif label["task"] == "event":
                            if label["type"] == "lm" and len(span) == 2:
                                pred_trigger_to_event[span[1]]["event_type"] = label["string"]  # fmt: skip
                            elif label["type"] == "lr" and len(span) == 3:
                                arg = self._convert_span_to_string(
                                    span[2], ins["input_ids"], self.transform.tokenizer
                                )
                                pred_trigger_to_event[span[1]]["arguments"].append(
                                    {"argument": arg, "role": label["string"]}
                                )
                        elif label["task"] == "discontinuous_ent" and len(span) > 1:
                            parts = [
                                self._convert_span_to_string(
                                    part, ins["input_ids"], self.transform.tokenizer
                                )
                                for part in span[1:]
                            ]
                            string = " ".join([part[0] for part in parts])
                            position = []
                            for part in parts:
                                position.append(part[1])
                            pred_discon_ents.append(
                                (label["string"], string, self.reset_position(position))
                            )
                        elif label["task"] == "hyper_rel" and len(span) == 5 and span[3] in span_to_label:  # fmt: skip
                            q_label = span_to_label[span[3]]
                            span_1 = self._convert_span_to_string(
                                span[1], ins["input_ids"], self.transform.tokenizer
                            )
                            span_2 = self._convert_span_to_string(
                                span[2], ins["input_ids"], self.transform.tokenizer
                            )
                            span_4 = self._convert_span_to_string(
                                span[4], ins["input_ids"], self.transform.tokenizer
                            )
                            pred_hyper_rels.append((label["string"], span_1, span_2, q_label["string"], span_4))  # fmt: skip
                    else:
                        # span task has no labels
                        pred_token_ids = []
                        for part in span:
                            _pred_token_ids = [ins["input_ids"][i] for i in part]
                            pred_token_ids.extend(_pred_token_ids)
                        span_string = self.transform.tokenizer.decode(pred_token_ids)
                        pred_spans.append(
                            (
                                span_string,
                                tuple(
                                    [
                                        tuple(
                                            self.reset_position(
                                                ins["input_ids"].cpu().tolist(), part
                                            )
                                        )
                                        for part in span
                                    ]
                                ),
                            )
                        )
                for trigger, item in pred_trigger_to_event.items():
                    trigger = self._convert_span_to_string(
                        trigger, ins["input_ids"], self.transform.tokenizer
                    )
                    if item["event_type"] not in raw_schema["event"]:
                        continue
                    legal_roles = raw_schema["event"][item["event_type"]]
                    pred_events.append(
                        {
                            "trigger": trigger,
                            "event_type": item["event_type"],
                            "arguments": [
                                arg
                                for arg in filter(
                                    lambda arg: arg["role"] in legal_roles,
                                    item["arguments"],
                                )
                            ],
                        }
                    )
                results.append(
                    {
                        "id": ins["raw"]["id"],
                        "results": {
                            "cls": pred_clses,
                            "ent": pred_ents,
                            "rel": pred_rels,
                            "event": pred_events,
                            "span": pred_spans,
                            "discon_ent": pred_discon_ents,
                            "hyper_rel": pred_hyper_rels,
                        },
                    }
                )

        return results
    def train(self):
        # dist.init_process_group(backend='nccl')
        # local_rank = dist.get_rank()
        if self.config.skip_train:
            raise RuntimeError(
                "Training procedure started while config.skip_train is True!"
            )
        else:
            logger.debug("Init optimizer")
            self.optimizer = self.init_optimizer()
            logger.debug(f"optimizer: {self.optimizer}")
            logger.debug("Prepare optimizer")
            self.optimizer = accelerator.prepare_optimizer(self.optimizer)
            logger.debug("Init lr_scheduler")
            self.lr_scheduler = self.init_lr_scheduler()
            if self.lr_scheduler is not None:
                logger.debug(f"lr_scheduler: {type(self.lr_scheduler)}")
                logger.debug("Prepare lr_scheduler")
                self.lr_scheduler = accelerator.prepare_scheduler(self.lr_scheduler)

        if self.config.resumed_training_path is not None:
            self.load(
                self.config.resumed_training_path,
                load_config=False,
                load_model=True,
                load_optimizer=True,
                load_history=True,
            )
            resumed_training = True
        else:
            resumed_training = False
            
        train_loader = self.get_data_loader("train", False, self.history["curr_epoch"])
        total_steps = self.history["curr_epoch"] * len(train_loader)
        start_time = datetime.now()
        hidden_prmopt = None
        for epoch_idx in range(self.history["curr_epoch"], self.config.num_epochs):

            logger.info(f"Start training {epoch_idx}/{self.config.num_epochs}")
            if not resumed_training:
                self.history["curr_epoch"] = epoch_idx

            used_time = datetime.now() - start_time
            time_per_epoch = safe_division(used_time, self.history["curr_epoch"])
            remain_time = time_per_epoch * (
                self.config.num_epochs - self.history["curr_epoch"]
            )
            logger.info(
                f"Epoch: {epoch_idx}/{self.config.num_epochs} [{str(used_time)}<{str(remain_time)}, {str(time_per_epoch)}/epoch]"
            )

            self.model.train()
            self.optimizer.zero_grad()
            epoch_train_loader = self.get_data_loader(
                "train", is_eval=False, epoch=epoch_idx
            )
            loader = pbar(epoch_train_loader, desc=f"Train(e{epoch_idx})")
            for batch_idx, batch in enumerate(loader):
                with accelerator.accumulate(self.model):
                    if not resumed_training:
                        self.history["curr_batch"] = batch_idx
                        self.history["total_steps"] = total_steps
                    if resumed_training and total_steps < self.history["total_steps"]:
                        total_steps += 1
                        continue
                    elif (
                        resumed_training and total_steps == self.history["total_steps"]
                    ):
                        resumed_training = False
                    #1.get the input from batch
                    #2.return the hidden_[Mask] vector respectively
                    #3.inject to the input
                    
                    # with torch.no_grad():
                    #     if hidden_prmopt is None:
                    #         prompts_ids = batch['prompts_ids']
                    #         prompts_mask = batch['prompts_mask']
                    #         hidden_prmopt = self.model.get_prompt_hidden(prompts_ids[0],prompts_mask[0])
                    #     batch['hidden_prmopt'] = hidden_prmopt.detach()
                        # torch.cuda.empty_cache()
                    
                    
                    result = self.model(**batch)
                    # self.get_error_sample_and_save(batch,result['pred'],self.instruction_path)
                    # self.convert_error_2_Instruction(error_results,'123')
                    # result["loss"] /= self.config.grad_accum_steps
                    accelerator.backward(result["loss"])
                    loss_item = result["loss"].item()
                    self.history["current_train_loss"]["epoch"] += loss_item
                    self.history["current_train_loss"]["step"] += loss_item
                    loader.set_postfix({"loss": loss_item})
                    self.log_loss(
                        self.history["total_steps"], loss_item, "step", "train"
                    )

                    if self.config.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.config.max_grad_norm
                        )
                    # if ((batch_idx + 1) % self.config.grad_accum_steps) == 0 or (
                    #     batch_idx + 1
                    # ) == len(loader):
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    if (
                        self.config.step_eval_interval > 0
                        and (self.history["total_steps"] + 1)
                        % self.config.step_eval_interval
                        == 0 and epoch_idx>2
                    ):
                        self._eval_during_train("step")
                        if not self._check_patience():
                            break
                    total_steps += 1
                    if not resumed_training:
                        self.history["total_steps"] += 1

            logger.info(loader)
            if (self.config.epoch_eval_interval > 0) and (
                ((epoch_idx + 1) % self.config.epoch_eval_interval) == 0 and epoch_idx>2
            ):
                if epoch_idx > self.config.num_epochs *0.1:
                    self._eval_during_train("epoch")
                    if not self._check_patience():
                        break

        logger.info("Trial finished.")
        if self.config.select_best_by_key == "metric":
            tmp_string = f"{self.history['best_metric']:.5f}"
        else:
            tmp_string = f"{self.history['best_loss']}"
        logger.info(
            f"Best epoch: {self.history['best_epoch']}, step: {self.history['best_step']}"
        )
        logger.info(
            f"Best {self.config.select_best_on_data}.{self.config.select_best_by_key}.{self.config.best_metric_field} : {tmp_string}"
        )

        if self.config.final_eval_on_test:
            logger.info("Loading best ckpt")
            self.load_best_ckpt()
            test_loss, test_measures = self.eval(
                "test", verbose=True, dump=True, postfix="final"
            )
            self.log_loss(0, test_loss, "final", "test")
            self.log_metrics(0, test_measures, "final", "test")
            return test_loss, test_measures

        self.after_whole_train()

        return self.history["best_loss"], self.history["best_metric"]
    from typing import Tuple
    @torch.no_grad()
    def eval(
        self, dataset_name, verbose=False, dump=False, dump_middle=False, postfix=""
    ) -> Tuple[float, dict]:
        from rex.utils.dict import get_dict_content
        from rex.utils.io import dump_json, dump_jsonlines
        """Eval on specific dataset and return loss and measurements

        Args:
            dataset_name: which dataset to evaluate
            verbose: whether to log evaluation results
            dump: if True, dump metric results to `self.measures_path`
            dump_middle: if True, dump middle results to `self.middle_path`
            postfix: filepath postfix for dumping

        Returns:
            eval_loss: float
            metrics: dict
        """
        self.model.eval()
        eval_loader = self.get_data_loader(
            dataset_name, is_eval=True, epoch=self.history["curr_epoch"]
        )
        loader = pbar(eval_loader, desc=f"{dataset_name} - {postfix} Eval", ascii=True)

        eval_loss = 0.0
        tot_batch_results = []
        hidden_prmopt = None
        for batch in loader:
            # with torch.no_grad():
            #     if hidden_prmopt is None:
            #         prompts_ids = batch['prompts_ids']
            #         prompts_mask = batch['prompts_mask']
            #         hidden_prmopt = self.model.get_prompt_hidden(prompts_ids[0],prompts_mask[0])
            #         torch.cuda.empty_cache()
            # batch['hidden_prmopt'] = hidden_prmopt.detach()

            out = self.model(**batch, is_eval=True)

            eval_loss += out["loss"].item()
            if 'hidden_prmopt' in batch:
                del batch['hidden_prmopt']
            batch_results: dict = self.metric(batch, out)
            batch_metric_score = get_dict_content(
                batch_results["metric_scores"], self.config.best_metric_field
            )
            loader.set_postfix({self.config.best_metric_field: batch_metric_score})

            batch_instances = [
                {"gold": gold, "pred": pred}
                for gold, pred in zip(batch_results["gold"], batch_results["pred"])
            ]
            tot_batch_results.extend(batch_instances)

        logger.info(loader)
        measurements = self.metric.compute()

        if verbose:
            logger.info(f"Eval dataset: {dataset_name}")
            logger.info(f"Eval loss: {eval_loss}")
            logger.info(
                f"Eval metrics: {get_dict_content(measurements, self.config.best_metric_field)}"
            )
        _filename_prefix = (
            f"{dataset_name}.{postfix}" if len(postfix) > 0 else f"{dataset_name}"
        )
        if dump:
            dump_obj = {
                "dataset_name": dataset_name,
                "eval_loss": eval_loss,
                "metrics": measurements,
            }
            _measure_result_filepath = self.measures_path.joinpath(
                f"{_filename_prefix}.json"
            )
            dump_json(dump_obj, _measure_result_filepath)
            logger.info(f"Dump measure results into {_measure_result_filepath}")
        if dump_middle:
            _middle_result_filepath = self.middle_path.joinpath(
                f"{_filename_prefix}.jsonl"
            )
            dump_jsonlines(tot_batch_results, _middle_result_filepath)
            logger.info(f"Dump middle results into {_middle_result_filepath}")

        self.metric.reset()

        return eval_loss, measurements


if __name__ == "__main__":
    pass
    # further_finetune()

    # from rex.utils.config import ConfigParser

    # config = ConfigParser.parse_cmd(cmd_args=["-dc", "conf/ner.yaml"])
    # config = ConfigParser.parse_cmd(cmd_args=["-dc", "conf/mirror-ace05en.yaml"])

    # task = MrcTaggingTask(
    #     config,
    #     initialize=True,
    #     makedirs=True,
    #     dump_configfile=True,
    # )
    # task = SchemaGuidedInstructBertTask.from_taskdir(
    #     "outputs/InstructBert_TagSpan_DebertaV3Base_ACE05EN_Rel",
    #     initialize=True,
    #     load_config=True,
    #     dump_configfile=False,
    # )
    # task = SchemaGuidedInstructBertTask(
    #     config,
    #     initialize=True,
    #     makedirs=True,
    #     dump_configfile=False,
    # )
    # task.load(
    #     "outputs/InstructBert_TagSpan_DebertaV3Base_ACE05EN_NerRelEvent/ckpt/SchemaGuidedInstructBertModel.epoch.0.pth",
    #     load_config=False,
    # )
    # task.eval("test", verbose=True, dump=True, dump_middle=True, postfix="re_eval")
    # task.load(
    #     # "outputs/Mirror_RobertaBaseWwm_Cons_MsraMrc/ckpt/MrcGlobalPointerModel.best.pth",
    #     # "outputs/Mirror_RobertaBaseWwm_W2_MsraMrc_HyperParamExp1/ckpt/MrcGlobalPointerModel.best.pth",
    #     config.base_model_path,
    #     load_config=False,
    #     load_model=True,
    #     load_optimizer=False,
    #     load_history=False,
    # )
    # task.train()
    # task = MrcTaggingTask.from_taskdir(
    #     "outputs/Mirror_W2_MSRAv2_NER",
    #     initialize=True,
    #     dump_configfile=False,
    #     load_config=True,
    # )
    # for name, _ in task.model.named_parameters():
    #     print(name)
    # task.eval("test", verbose=True, dump=True, dump_middle=True, postfix="re_eval.0.1")

    # task = MrcQaTask(
    #     config,
    #     initialize=True,
    #     makedirs=True,
    #     dump_configfile=True,
    # )
    # task.train()
    # task.eval("dev", verbose=True, dump=True, dump_middle=True, postfix="re_eval")
