"""Minimal LongBench evaluation entry for GrSlerp/Qwen models.

Example:
    python GrSlerp/eval_longbench.py \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --dataset qasper hotpotqa \
        --use-grslerp \
        --compression-ratio 0.375 \
        --recent-ratio 0.1 \
        --start-keep 1
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional runtime dependency
    load_dataset = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional runtime dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None

from modeling_qwen_grslerp_v5 import convert_kvcache_qwen2_grslerp_v5


DEFAULT_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

DEFAULT_E_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase "
        "if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "qasper": (
        "You are given a scientific article and a question.\n"
        "Answer the question as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write "
        '"unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". '
        "Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase "
        "or sentence if possible.\n"
        'If the question cannot be answered based on the information in the article, write "unanswerable". '
        'If the question is a yes/no question, answer "yes", "no", or "unanswerable".\n'
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not "
        "output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并用中文简短回答：\n\n"
        "{context}\n\n"
        "现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    "dureader": (
        "请基于给定的文章回答下述问题。\n\n"
        "文章：{context}\n\n"
        "请基于上述文章回答下面的问题。\n\n"
        "问题：{input}\n"
        "回答："
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n"
        "{context}\n\n"
        "Now, write a one-page summary of the report.\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction.\n"
        "Answer the query in one or more sentences.\n\n"
        "Transcript:\n"
        "{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\n"
        "Answer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\n"
        "News:\n"
        "{context}\n\n"
        "Now, write a one-page summary of all the news.\n\n"
        "Summary:"
    ),
    "vcsum": (
        "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n"
        "会议记录：\n"
        "{context}\n\n"
        "会议总结："
    ),
    "trec": (
        "Please determine the type of the question below.\n"
        "Here are some examples of questions.\n\n"
        "{context}\n"
        "{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any "
        "other words. The following are some examples.\n\n"
        "{context}\n\n"
        "{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n"
        "{input}"
    ),
    "lsht": (
        "请判断给定新闻的类别，下面是一些例子。\n\n"
        "{context}\n"
        "{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia.\n"
        "Some of them may be duplicates. Please carefully read these paragraphs and determine how many "
        "unique paragraphs there are after removing duplicates. In other words, how many non-repeating "
        "paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. The output format "
        "should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract.\n"
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n"
        "{input}\n\n"
        'Please enter the number of the paragraph that the abstract is from. The answer format must be '
        'like "Paragraph 1", "Paragraph 2", etc.\n\n'
        "The answer is: "
    ),
    "passage_retrieval_zh": (
        "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n"
        "{context}\n\n"
        "下面是一个摘要\n\n"
        "{input}\n\n"
        '请输入摘要所属段落的编号。答案格式必须是"段落1"、"段落2"等格式。\n\n'
        "答案是："
    ),
    "lcc": (
        "Please complete the code given below.\n\n"
        "{context}"
        "Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below.\n"
        "{context}"
        "{input}"
        "Next line of code:\n"
    ),
}

DATASET2MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

NO_CHAT_TEMPLATE_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}

FIRST_LINE_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple LongBench evaluator for GrSlerp/Qwen models.")
    parser.add_argument("--model-path", type=str, required=True, help="Hugging Face model path or local path.")
    parser.add_argument("--dataset", nargs="*", default=None, help="Datasets to evaluate. Default: all.")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E.")
    parser.add_argument(
        "--hf-dataset-repo",
        type=str,
        default="THUDM/LongBench",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional local dataset directory. Expects <dataset>.jsonl or <dataset>_e.jsonl.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for predictions and scores.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output sub-directory name.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, cpu ...")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Override model context length used for middle truncation.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attention implementation passed to transformers, e.g. eager/sdpa/flash_attention_2.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True.")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate only the first N samples per dataset.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dataset evaluation when the prediction file already exists.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Do not wrap prompts with tokenizer chat template.",
    )
    parser.add_argument("--use-grslerp", action="store_true", help="Enable GrSlerp-v5 KV cache compression.")
    parser.add_argument("--compression-ratio", type=float, default=0.375, help="GrSlerp compression ratio.")
    parser.add_argument("--recent-ratio", type=float, default=0.1, help="GrSlerp recent window ratio.")
    parser.add_argument("--start-keep", type=int, default=1, help="GrSlerp prefix keep budget.")
    parser.add_argument(
        "--schedule-type",
        type=str,
        choices=["linear", "cosine", "flat"],
        default="linear",
        help="Layer compression schedule.",
    )
    return parser.parse_args()


def ensure_runtime_dependencies() -> None:
    missing = []
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        missing.append("transformers")
    if missing:
        raise RuntimeError(
            "Missing required dependency: "
            + ", ".join(missing)
            + ". Please install it before running LongBench evaluation."
        )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def infer_context_window(
    model: Any,
    tokenizer: Any,
    override_max_context_tokens: Optional[int],
) -> int:
    if override_max_context_tokens is not None:
        return override_max_context_tokens

    candidates = []
    for value in (
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "max_sequence_length", None),
        getattr(model.config, "sliding_window", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)

    if candidates:
        return max(candidates)
    return 8192


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def get_output_root(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        base_dir = Path(args.output_dir)
    else:
        parent = Path(__file__).resolve().parent
        base_dir = parent / ("pred_e" if args.e else "pred")

    run_name = args.run_name
    if run_name is None:
        model_name = sanitize_name(Path(args.model_path).name or args.model_path)
        suffix = "-grslerp-v5" if args.use_grslerp else "-base"
        run_name = model_name + suffix
    return base_dir / run_name


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_longbench_examples(
    dataset_name: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    resolved_name = f"{dataset_name}_e" if args.e else dataset_name
    if args.data_root is not None:
        data_path = Path(args.data_root) / f"{resolved_name}.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"Local dataset file not found: {data_path}")
        rows = load_jsonl(data_path)
    else:
        if load_dataset is None:
            raise RuntimeError(
                "datasets is not installed, and --data-root was not provided. "
                "Please install datasets or point to a local jsonl directory."
            )
        rows = [dict(row) for row in load_dataset(args.hf_dataset_repo, resolved_name, split=args.split)]

    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    return rows


def build_prompt(example: Dict[str, Any], dataset_name: str) -> str:
    prompt_template = DATASET2PROMPT[dataset_name]
    return prompt_template.format(**example)


def maybe_apply_chat_template(
    prompt: str,
    tokenizer: Any,
    dataset_name: str,
    disable_chat_template: bool,
) -> str:
    if disable_chat_template or dataset_name in NO_CHAT_TEMPLATE_DATASETS:
        return prompt
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def truncate_from_middle(
    input_ids: torch.Tensor,
    max_prompt_tokens: int,
) -> torch.Tensor:
    if input_ids.shape[-1] <= max_prompt_tokens:
        return input_ids
    left = max_prompt_tokens // 2
    right = max_prompt_tokens - left
    return torch.cat([input_ids[:left], input_ids[-right:]], dim=0)


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def normalize_zh_answer(text: str) -> str:
    cn_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
        "｟｠｢｣､、〃》「」『』〖〗〔〕〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    )
    all_punctuation = set(string.punctuation + cn_punctuation)
    return "".join(ch for ch in text.lower() if ch not in all_punctuation and not ch.isspace())


def lcs_length(xs: Sequence[str], ys: Sequence[str]) -> int:
    if not xs or not ys:
        return 0
    dp = [0] * (len(ys) + 1)
    for x in xs:
        prev = 0
        for idx, y in enumerate(ys, start=1):
            old = dp[idx]
            if x == y:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = old
    return dp[-1]


def rouge_l_score(prediction: str, ground_truth: str, zh: bool = False) -> float:
    if zh:
        pred_tokens = list(normalize_zh_answer(prediction))
        gt_tokens = list(normalize_zh_answer(ground_truth))
    else:
        pred_tokens = normalize_answer(prediction).split()
        gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gt_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def token_f1(prediction_tokens: Sequence[str], ground_truth_tokens: Sequence[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    return token_f1(pred_tokens, gt_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = list(normalize_zh_answer(prediction))
    gt_tokens = list(normalize_zh_answer(ground_truth))
    if not pred_tokens or not gt_tokens:
        return 0.0
    return token_f1(pred_tokens, gt_tokens)


def classification_score(prediction: str, ground_truth: str, all_classes: Optional[Sequence[str]]) -> float:
    if not all_classes:
        return 1.0 if ground_truth in prediction else 0.0
    matches = [class_name for class_name in all_classes if class_name in prediction]
    filtered = []
    for class_name in matches:
        if class_name in ground_truth and class_name != ground_truth:
            continue
        filtered.append(class_name)
    if ground_truth in filtered:
        return 1.0 / len(filtered)
    return 0.0


def retrieval_score(prediction: str, ground_truth: str, zh: bool = False) -> float:
    pattern = r"段落(\d+)" if zh else r"Paragraph (\d+)"
    match = re.findall(pattern, ground_truth)
    if not match:
        return 0.0
    target_id = match[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    hits = sum(1 for number in numbers if number == target_id)
    return hits / len(numbers)


def count_score(prediction: str, ground_truth: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    hits = sum(1 for number in numbers if number == str(ground_truth))
    return hits / len(numbers)


def code_sim_score(prediction: str, ground_truth: str) -> float:
    import difflib

    cleaned_prediction = ""
    for line in prediction.lstrip("\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("`", "#", "//")):
            continue
        cleaned_prediction = stripped
        break
    if not cleaned_prediction:
        cleaned_prediction = prediction.strip()
    return difflib.SequenceMatcher(a=cleaned_prediction, b=ground_truth).ratio()


def score_single_prediction(
    dataset_name: str,
    prediction: str,
    answers: Sequence[str],
    all_classes: Optional[Sequence[str]] = None,
) -> float:
    if dataset_name in FIRST_LINE_DATASETS:
        prediction = prediction.lstrip("\n").split("\n")[0]

    best_score = 0.0
    for answer in answers:
        if dataset_name in {
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "triviaqa",
        }:
            score = qa_f1_score(prediction, answer)
        elif dataset_name in {"multifieldqa_zh"}:
            score = qa_f1_zh_score(prediction, answer)
        elif dataset_name in {"dureader", "vcsum"}:
            score = rouge_l_score(prediction, answer, zh=True)
        elif dataset_name in {"gov_report", "qmsum", "multi_news", "samsum"}:
            score = rouge_l_score(prediction, answer, zh=False)
        elif dataset_name in {"trec", "lsht"}:
            score = classification_score(prediction, answer, all_classes)
        elif dataset_name in {"passage_retrieval_en"}:
            score = retrieval_score(prediction, answer, zh=False)
        elif dataset_name in {"passage_retrieval_zh"}:
            score = retrieval_score(prediction, answer, zh=True)
        elif dataset_name in {"passage_count"}:
            score = count_score(prediction, answer)
        elif dataset_name in {"lcc", "repobench-p"}:
            score = code_sim_score(prediction, answer)
        else:
            raise ValueError(f"Unsupported dataset for scoring: {dataset_name}")
        best_score = max(best_score, score)
    return best_score


def score_dataset(
    dataset_name: str,
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not rows:
        return {"score": 0.0, "num_samples": 0}

    scores = []
    bucket_scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for row in rows:
        answers = row.get("answers") or []
        if not isinstance(answers, list):
            answers = [answers]
        score = score_single_prediction(
            dataset_name=dataset_name,
            prediction=row["pred"],
            answers=answers,
            all_classes=row.get("all_classes"),
        )
        scores.append(score)

        length = row.get("length")
        if isinstance(length, (int, float)):
            if length < 4000:
                bucket_scores["0-4k"].append(score)
            elif length < 8000:
                bucket_scores["4-8k"].append(score)
            else:
                bucket_scores["8k+"].append(score)

    result = {
        "score": round(100.0 * sum(scores) / len(scores), 2),
        "num_samples": len(scores),
    }

    if any(bucket_scores.values()):
        result["length_buckets"] = {
            name: round(100.0 * sum(values) / len(values), 2)
            for name, values in bucket_scores.items()
            if values
        }
    return result


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, torch.device, int]:
    ensure_runtime_dependencies()

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    effective_attn_impl = "eager" if args.use_grslerp else args.attn_implementation
    if effective_attn_impl is not None:
        model_kwargs["attn_implementation"] = effective_attn_impl

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    model.to(device)
    model.eval()

    if args.use_grslerp:
        grslerp_cfg = SimpleNamespace(
            compression_ratio=args.compression_ratio,
            recent_ratio=args.recent_ratio,
            start_keep=args.start_keep,
        )
        try:
            convert_kvcache_qwen2_grslerp_v5(
                model,
                grslerp_cfg,
                schedule_type=args.schedule_type,
                force_output_attentions=True,
            )
        except ValueError as exc:
            raise RuntimeError(f"Failed to enable GrSlerp-v5: {exc}") from exc

    max_context_tokens = infer_context_window(model, tokenizer, args.max_context_tokens)
    return model, tokenizer, device, max_context_tokens


def generate_prediction(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_context_tokens: int,
    dataset_name: str,
    example: Dict[str, Any],
    disable_chat_template: bool,
) -> str:
    prompt = build_prompt(example, dataset_name)
    prompt = maybe_apply_chat_template(prompt, tokenizer, dataset_name, disable_chat_template)

    max_new_tokens = DATASET2MAX_NEW_TOKENS[dataset_name]
    max_prompt_tokens = max(1, max_context_tokens - max_new_tokens)

    tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = tokenized["input_ids"][0]
    input_ids = truncate_from_middle(input_ids, max_prompt_tokens=max_prompt_tokens)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    generated_ids = output[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_dataset(
    dataset_name: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_context_tokens: int,
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    pred_path = output_root / f"{dataset_name}.jsonl"

    if args.skip_existing and pred_path.exists():
        cached_rows = load_jsonl(pred_path)
        result = score_dataset(dataset_name, cached_rows)
        print(f"[skip] {dataset_name}: {result['score']:.2f}")
        return result

    examples = load_longbench_examples(dataset_name, args)
    written_rows: List[Dict[str, Any]] = []

    with pred_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(examples, start=1):
            pred = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_context_tokens=max_context_tokens,
                dataset_name=dataset_name,
                example=example,
                disable_chat_template=args.disable_chat_template,
            )

            row = {
                "_id": example.get("_id", idx),
                "pred": pred,
                "answers": example.get("answers"),
                "all_classes": example.get("all_classes"),
                "length": example.get("length"),
            }
            written_rows.append(row)
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

            if idx % 10 == 0 or idx == len(examples):
                print(f"[{dataset_name}] {idx}/{len(examples)} done", flush=True)

    result = score_dataset(dataset_name, written_rows)
    print(f"[done] {dataset_name}: {result['score']:.2f}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    datasets = args.dataset or (DEFAULT_E_DATASETS if args.e else DEFAULT_DATASETS)

    unknown_datasets = [name for name in datasets if name not in DATASET2PROMPT]
    if unknown_datasets:
        raise ValueError(f"Unsupported dataset(s): {unknown_datasets}")

    model, tokenizer, device, max_context_tokens = load_model_and_tokenizer(args)
    output_root = get_output_root(args)

    print(f"model_path={args.model_path}")
    print(f"device={device}")
    print(f"dtype={resolve_dtype(args.dtype, device)}")
    print(f"max_context_tokens={max_context_tokens}")
    print(f"use_grslerp={args.use_grslerp}")
    print(f"output_dir={output_root}")

    all_results: Dict[str, Any] = {}
    summary_scores = []
    for dataset_name in datasets:
        result = evaluate_dataset(
            dataset_name=dataset_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context_tokens=max_context_tokens,
            args=args,
            output_root=output_root,
        )
        all_results[dataset_name] = result
        summary_scores.append(result["score"])

    if summary_scores:
        all_results["average"] = round(sum(summary_scores) / len(summary_scores), 2)

    result_path = output_root / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nSummary")
    for dataset_name in datasets:
        print(f"  {dataset_name:>20s}: {all_results[dataset_name]['score']:.2f}")
    if "average" in all_results:
        print(f"  {'average':>20s}: {all_results['average']:.2f}")
    print(f"\nSaved results to {result_path}")


if __name__ == "__main__":
    main()
