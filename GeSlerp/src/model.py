import torch
import torch.nn as nn
import os
from rex.utils.iteration import windowed_queue_iter
from transformers import AutoModel, BertModel,AutoConfig
from torchvision.models.vision_transformer import VisionTransformer
from src.utils import decode_nnw_nsw_thw_mat, decode_nnw_thw_mat, decode_pointer_mat
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from transformers.models.deberta_v2 import modeling_deberta_v2
# from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model
from.base_models import Prompt_Fusion_Model
from collections import OrderedDict
import json
from .config import Prompt_Fusion_Config


def normalize(v):
    norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    return v / norm, norm


def slerp_for_batch_hidden(p, q, t):
    # 假设 p, q 的形状为 (bs, seq, dim)
    # t 的形状应为 (bs, seq, 1) 或者可以广播至此形状
    p,norm1 = normalize(p)
    q,norm2 = normalize(q)
    # 确保t可以广播到p和q的形状上
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    
    # 计算p和q之间的点积
    dot = torch.sum(p * q, dim=-1, keepdim=True)  # 形状为 (bs, num_head, seq, 1)

    # 限制在数值稳定的范围内
    dot = torch.clamp(dot, -1.0, 1.0)

    # 计算角度theta
    theta = torch.acos(dot)

    # 三角函数操作
    sin_theta = torch.sin(theta)
    sin_theta_t = torch.sin(t * theta)
    sin_theta_1_t = torch.sin((1 - t) * theta)
    
    # 避免除以0的情况，当sin(theta)接近0时（即p和q非常接近），slerp退化为线性插值
    slerp_t = torch.where(sin_theta > 1e-6,
                          (sin_theta_1_t * p + sin_theta_t * q) / sin_theta,
                          (1 - t) * p + t * q)
    norm_result = (1.0 - t) * norm1 + t * norm2
    slerp_t = slerp_t * norm_result
    return slerp_t

def slerp_for_batch(p, q, t):
    # 假设 p, q 的形状为 (bs, num_head, seq, dim)
    # t 的形状应为 (bs, num_head, seq, 1) 或者可以广播至此形状
    p,norm1 = normalize(p)
    q,norm2 = normalize(q)
    # 确保t可以广播到p和q的形状上
    if t.dim() == 3:
        t = t.unsqueeze(-1)
    
    # 计算p和q之间的点积
    dot = torch.sum(p * q, dim=-1, keepdim=True)  # 形状为 (bs, num_head, seq, 1)

    # 限制在数值稳定的范围内
    dot = torch.clamp(dot, -1.0, 1.0)

    # 计算角度theta
    theta = torch.acos(dot)

    # 三角函数操作
    sin_theta = torch.sin(theta)
    sin_theta_t = torch.sin(t * theta)
    sin_theta_1_t = torch.sin((1 - t) * theta)
    
    # 避免除以0的情况，当sin(theta)接近0时（即p和q非常接近），slerp退化为线性插值
    slerp_t = torch.where(sin_theta > 1e-6,
                          (sin_theta_1_t * p + sin_theta_t * q) / sin_theta,
                          (1 - t) * p + t * q)
    norm_result = (1.0 - t) * norm1 + t * norm2
    slerp_t = slerp_t * norm_result
    return slerp_t

class Biaffine(nn.Module):
    """Biaffine transformation

    References:
        - https://github.com/yzhangcs/parser/blob/main/supar/modules/affine.py
        - https://github.com/ljynlp/W2NER
    """

    def __init__(self, n_in, n_out=2, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros(n_out, n_in + int(bias_x), n_in + int(bias_y))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        # s = s.permute(0, 2, 3, 1)

        return s


class LinearWithAct(nn.Module):
    def __init__(self, n_in, n_out, dropout=0) -> None:
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, heads=8, dropout=0.1):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(hidden_size, heads, dropout=dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # 调整维度以符合MultiheadAttention的输入要求
        attn_output, _ = self.multi_head_attn(x, x, x)
        return attn_output.permute(1, 0, 2)

class PointerMatrix(nn.Module):
    """Pointer Matrix Prediction with Multi-layer Self Attention"""

    def __init__(
        self,
        hidden_size,
        biaffine_size,
        cls_num=2,
        dropout=0,
        biaffine_bias=False,
        use_rope=False,
        num_self_attn_layers=2  # 新增参数：自注意力层数
    ):
        super().__init__()
        self.linear_h = LinearWithAct(
            n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        )
        self.linear_t = LinearWithAct(
            n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        )
        self.biaffine = Biaffine(
            n_in=biaffine_size,
            n_out=cls_num,
            bias_x=biaffine_bias,
            bias_y=biaffine_bias,
        )
        # 创建一个ModuleList来存储多个自注意力层
        # self.self_attn_layers = nn.ModuleList([
        #     SelfAttention(hidden_size=biaffine_size)
        #     for _ in range(num_self_attn_layers)
        # ])
        self.use_rope = use_rope
    def sinusoidal_position_embedding(self, qw, kw):
        batch_size, seq_len, output_dim = qw.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        pos_emb = position_ids * indices
        pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1)
        pos_emb = pos_emb.repeat((batch_size, *([1] * len(pos_emb.shape))))
        pos_emb = torch.reshape(pos_emb, (batch_size, seq_len, output_dim))
        pos_emb = pos_emb.to(qw)

        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.cat([-qw[..., 1::2], qw[..., ::2]], -1)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.cat([-kw[..., 1::2], kw[..., ::2]], -1)
        kw = kw * cos_pos + kw2 * sin_pos
        return qw, kw
    def forward(self, x):
        h = self.linear_h(x)
        t = self.linear_t(x)
        # 依次应用每一层自注意力
        # for self_attn in self.self_attn_layers:
        #     h_temp = self_attn(h)
        #     t_temp = self_attn(t)
        if self.use_rope:
            h, t = self.sinusoidal_position_embedding(h, t)
        o = self.biaffine(h, t)
        return o

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()
        # 创建一个足够长的位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(max_len, hidden_size))
        nn.init.normal_(self.pos_encoding)

    def forward(self, x):
        length = x.size(1)
        return self.pos_encoding[:length, :]



def multilabel_categorical_crossentropy(y_pred, y_true, bit_mask=None):
    """
    https://kexue.fm/archives/7359
    https://github.com/gaohongkui/GlobalPointer_pytorch/blob/main/common/utils.py
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    if bit_mask is None:
        return neg_loss + pos_loss
    else:
        raise NotImplementedError


class MrcPointerMatrixModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        cls_num: int = 2,
        biaffine_size: int = 384,
        none_type_id: int = 0,
        text_mask_id: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # num of predicted classes, default is 3: None, NNW and THW
        self.cls_num = cls_num
        # None type id: 0, Next Neighboring Word (NNW): 1, Tail Head Word (THW): 2
        self.none_type_id = none_type_id
        # input: cls instruction sep text sep pad
        # mask:   1       2       3   4    5   0
        self.text_mask_id = text_mask_id

        self.plm = BertModel.from_pretrained(plm_dir)
        hidden_size = self.plm.config.hidden_size
        # self.biaffine_size = biaffine_size
        self.nnw_mat = PointerMatrix(
            hidden_size, biaffine_size, cls_num=2, dropout=dropout
        )
        self.thw_mat = PointerMatrix(
            hidden_size, biaffine_size, cls_num=2, dropout=dropout
        )
        self.criterion = nn.CrossEntropyLoss()

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        mask_mat = (
            mask.eq(self.text_mask_id).unsqueeze(-1).expand((bs, seq_len, seq_len))
        )
        # bit_mask: (batch_size, seq_len, seq_len, 1)
        bit_mask = (
            torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).long()
        )
        return bit_mask

    def forward(self, input_ids, mask, labels=None, is_eval=False, **kwargs):
        hidden = self.input_encoding(input_ids, mask)
        nnw_hidden = self.nnw_mat(hidden)
        thw_hidden = self.thw_mat(hidden)
        # nnw_hidden = nnw_hidden / self.biaffine_size ** 0.5
        # thw_hidden = thw_hidden / self.biaffine_size ** 0.5
        # # (bs, 2, seq_len, seq_len)
        bs, _, seq_len, seq_len = nnw_hidden.shape

        bit_mask = self.build_bit_mask(mask)

        results = {"logits": {"nnw": nnw_hidden, "thw": thw_hidden}}
        if labels is not None:
            # mean
            nnw_loss = self.criterion(
                nnw_hidden.permute(0, 2, 3, 1).reshape(-1, 2),
                labels[:, 0, :, :].reshape(-1),
            )
            thw_loss = self.criterion(
                thw_hidden.permute(0, 2, 3, 1).reshape(-1, 2),
                labels[:, 1, :, :].reshape(-1),
            )
            loss = nnw_loss + thw_loss
            results["loss"] = loss

        if is_eval:
            batch_positions = self.decode(nnw_hidden, thw_hidden, bit_mask, **kwargs)
            results["pred"] = batch_positions
        return results

    def decode(
        self,
        nnw_hidden: torch.Tensor,
        thw_hidden: torch.Tensor,
        bit_mask: torch.Tensor,
        **kwargs,
    ):
        # B x L x L
        nnw_pred = nnw_hidden.argmax(1)
        thw_pred = thw_hidden.argmax(1)
        # B x 2 x L x L
        pred = torch.stack([nnw_pred, thw_pred], dim=1)
        pred = pred * bit_mask

        batch_preds = decode_nnw_thw_mat(pred, offsets=kwargs.get("offset"))

        return batch_preds


class MrcGlobalPointerModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        use_rope: bool = True,
        cls_num: int = 2,
        biaffine_size: int = 384,
        none_type_id: int = 0,
        text_mask_id: int = 4,
        dropout: float = 0.3,
        mode: str = "w2",
    ):
        super().__init__()

        # num of predicted classes, default is 3: None, NNW and THW
        self.cls_num = cls_num
        # None type id: 0, Next Neighboring Word (NNW): 1, Tail Head Word (THW): 2
        self.none_type_id = none_type_id
        # input: cls instruction sep text sep pad
        # mask:   1       2       3   4    5   0
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        # mode: w2: w2ner, cons: consecutive spans
        self.mode = mode
        assert self.mode in ["w2", "cons"]

        self.plm = BertModel.from_pretrained(plm_dir)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=2 if self.mode == "w2" else 1,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        mask_mat = (
            mask.eq(self.text_mask_id).unsqueeze(-1).expand((bs, seq_len, seq_len))
        )
        # bit_mask: (batch_size, 1, seq_len, seq_len)
        bit_mask = (
            torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        )
        if self.mode == "cons":
            bit_mask = bit_mask.triu()

        return bit_mask

    def forward(
        self, input_ids, mask, labels=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        # (bs, 2, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 2, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        assert labels.shape == (bs, cls_num, seq_len, seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss

        if is_eval:
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        assert self.mode in ["w2", "cons"]
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            if self.mode == "w2":
                for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                    path_prob *= probs[0, se[0], se[-1]]
                path_prob *= probs[1, path[-1], path[0]]
            elif self.mode == "cons":
                path_prob = probs[0, path[0], path[-1]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        **kwargs,
    ):
        # mode: w2: w2ner with nnw and thw labels, cons: consecutive spans with one type of labels
        assert self.mode in ["w2", "cons"]
        # B x 2 x L x L
        probs = logits.sigmoid()
        pred = (probs > top_p).long()
        if self.mode == "w2":
            preds = decode_nnw_thw_mat(pred, offsets=kwargs.get("offset"))
        elif self.mode == "cons":
            pred = pred.triu()
            preds = decode_pointer_mat(pred, offsets=kwargs.get("offset"))

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        return batch_preds


class SchemaGuidedInstructBertModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        vocab_size: int = None,
        use_rope: bool = True,
        biaffine_size: int = 512,
        label_mask_id: int = 4,
        text_mask_id: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        self.plm = AutoModel.from_pretrained(plm_dir)
        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        # _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        # mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # # bit_mask: (batch_size, 1, seq_len, seq_len)
        # bit_mask = (
        #     torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        # )
        bit_mask = (
            mask.gt(0).unsqueeze(1).unsqueeze(1).expand(bs, 1, seq_len, seq_len).float()
        )

        return bit_mask

    def forward(
        self, input_ids, mask, spans,labels=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        # (bs, 3, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        assert labels.shape == (bs, cls_num, seq_len, seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss
            # batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            # results["pred"] = batch_positions

        if is_eval:
            # pdb.set_trace()
            droplabels = torch.zeros((bs, 1, seq_len, seq_len)).to(logits)
            for i in range(bs):
                for span in spans[i]:
                    stolist = []
                    for t in span:
                        if len(t)==1: stolist.append(t[0])
                        else: 
                            stolist.extend([t[0],t[1]])
                    for m in stolist:
                        for n in stolist:
                            droplabels[i, 0, m, n] = 1
            droplabels = droplabels.repeat(1, cls_num, 1, 1)
            zeromask = droplabels != 0
            droplabels = droplabels * zeromask
            logits = logits * droplabels
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, spans=spans, **kwargs)
            results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        # legal_num_parts: tuple = (1, 2, 3),
        legal_num_parts: tuple = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # B x 3 x L x L
        if labels is None:
            # `labels` is used for upper bound analysis
            probs = logits.sigmoid()
            pred = (probs > top_p).long()
        else:
            pred = labels
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        # for pred, gold in zip(preds, kwargs.get("spans")):
        #     sorted_pred = sorted(set(tuple(x) for x in pred))
        #     sorted_gold = sorted(set(tuple(x) for x in gold))
        #     if sorted_pred != sorted_gold:
        #         breakpoint()

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        if legal_num_parts is not None:
            legal_preds = []
            for ins_paths in batch_preds:
                legal_paths = []
                for path in ins_paths:
                    if len(path) in legal_num_parts:
                        legal_paths.append(path)
                legal_preds.append(legal_paths)
        else:
            legal_preds = batch_preds

        return legal_preds



class SchemaGuidedInstructBertModelWithPromptReplace(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        vocab_size: int = None,
        use_rope: bool = True,
        biaffine_size: int = 512,
        label_mask_id: int = 4,
        text_mask_id: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        self.plm = AutoModel.from_pretrained(plm_dir)
        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        # _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        # mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # # bit_mask: (batch_size, 1, seq_len, seq_len)
        # bit_mask = (
        #     torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        # )
        bit_mask = (
            mask.gt(0).unsqueeze(1).unsqueeze(1).expand(bs, 1, seq_len, seq_len).float()
        )

        return bit_mask

    def prmopt_encoding(self,prompts_to_indices,hidden,prompts_ids, prompts_mask) ->torch.Tensor:
        #1.get the hidden_sentence_state from the last layer
        hidden_prmopt = self.input_encoding(prompts_ids,prompts_mask)
        hiddens = hidden_prmopt[:,-2,:] # remove [sep] and .
        #(optional) linear project to the embedding space
        #2.get the hidden_state corresponds to the prompt from the hidden vector
        for batch_index, indices in enumerate(prompts_to_indices):
            # 对于该batch中的每个prompt位置
            for prompt_index, hidden_index in enumerate(indices[:,-1]):
                # 获取对应的prompt隐藏状态
                prompt_hidden = hiddens[prompt_index]
                # 替换hidden中的对应位置
                hidden[batch_index, hidden_index] = (prompt_hidden+hidden[batch_index, hidden_index])/2
                # hidden[batch_index, hidden_index] += prompt_hidden
                # hidden[batch_index, hidden_index] = LinearWithAct(prompt_hidden)

        return hidden
    def forward(
        self, input_ids,prompts_to_indices,prompts_ids,mask,prompts_mask, labels=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        hidden = self.prmopt_encoding(prompts_to_indices,hidden,prompts_ids[0], prompts_mask[0])
        # (bs, 3, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        assert labels.shape == (bs, cls_num, seq_len, seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss
            # batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            # results["pred"] = batch_positions

        if is_eval:
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        # legal_num_parts: tuple = (1, 2, 3),
        legal_num_parts: tuple = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # B x 3 x L x L
        if labels is None:
            # `labels` is used for upper bound analysis
            probs = logits.sigmoid()
            pred = (probs > top_p).long()
        else:
            pred = labels
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        # for pred, gold in zip(preds, kwargs.get("spans")):
        #     sorted_pred = sorted(set(tuple(x) for x in pred))
        #     sorted_gold = sorted(set(tuple(x) for x in gold))
        #     if sorted_pred != sorted_gold:
        #         breakpoint()

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        if legal_num_parts is not None:
            legal_preds = []
            for ins_paths in batch_preds:
                legal_paths = []
                for path in ins_paths:
                    if len(path) in legal_num_parts:
                        legal_paths.append(path)
                legal_preds.append(legal_paths)
        else:
            legal_preds = batch_preds

        return legal_preds

class TrainableFusion(nn.Module):
    def __init__(self,fusion):
        super(TrainableFusion, self).__init__()
        self.weight = nn.Parameter(torch.tensor(fusion))

    def forward(self, tensor1, tensor2):
        return tensor1 * self.weight + tensor2 * (1 - self.weight)

class SchemaGuidedInstructBertModelWithPromptReplace_Concatenate(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        vocab_size: int = None,
        use_rope: bool = True,
        biaffine_size: int = 512,
        label_mask_id: int = 4,
        text_mask_id: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        self.plm = AutoModel.from_pretrained(plm_dir)
        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        self.fusion = TrainableFusion(fusion=0.5)
        self.linear_proj = LinearWithAct(n_in=self.plm.config.hidden_size*2,
                                         n_out=self.plm.config.hidden_size,
                                         dropout=0.3)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        # _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        # mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # # bit_mask: (batch_size, 1, seq_len, seq_len)
        # bit_mask = (
        #     torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        # )
        bit_mask = (
            mask.gt(0).unsqueeze(1).unsqueeze(1).expand(bs, 1, seq_len, seq_len).float()
        )

        return bit_mask

    def prmopt_encoding(self,prompts_to_indices,hidden,prompts_ids, prompts_mask) ->torch.Tensor:
        #1.get the hidden_sentence_state from the last layer
        hidden_prmopt = self.input_encoding(prompts_ids,prompts_mask)
        hiddens = hidden_prmopt[:,-2,:] # remove [sep] and .
        updated_hidden = hidden.clone()
        #(optional) linear project to the embedding space
        #2.get the hidden_state corresponds to the prompt from the hidden vector
        for batch_index, indices in enumerate(prompts_to_indices):
            # 对于该batch中的每个prompt位置
            for prompt_index, hidden_index in enumerate(indices[:,-1]):
                # 获取对应的prompt隐藏状态
                prompt_hidden = hiddens[prompt_index]
                # 替换hidden中的对应位置
                temp = torch.cat((hidden[batch_index, hidden_index],prompt_hidden),dim=0)
                prompt_hidden = self.linear_proj(temp)
                updated_hidden[batch_index, hidden_index] = self.fusion(prompt_hidden,hidden[batch_index, hidden_index])
                # hidden[batch_index, hidden_index] += prompt_hidden
                # hidden[batch_index, hidden_index] = LinearWithAct(prompt_hidden)
        return updated_hidden
    
    
    
    def forward(
        self, input_ids,prompts_to_indices,prompts_ids,mask,prompts_mask, labels=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        hidden = self.prmopt_encoding(prompts_to_indices,hidden,prompts_ids[0], prompts_mask[0])
        # (bs, 3, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        assert labels.shape == (bs, cls_num, seq_len, seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss
            # batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            # results["pred"] = batch_positions

        if is_eval:
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        # legal_num_parts: tuple = (1, 2, 3),
        legal_num_parts: tuple = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # B x 3 x L x L
        if labels is None:
            # `labels` is used for upper bound analysis
            probs = logits.sigmoid()
            pred = (probs > top_p).long()
        else:
            pred = labels
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        # for pred, gold in zip(preds, kwargs.get("spans")):
        #     sorted_pred = sorted(set(tuple(x) for x in pred))
        #     sorted_gold = sorted(set(tuple(x) for x in gold))
        #     if sorted_pred != sorted_gold:
        #         breakpoint()

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        if legal_num_parts is not None:
            legal_preds = []
            for ins_paths in batch_preds:
                legal_paths = []
                for path in ins_paths:
                    if len(path) in legal_num_parts:
                        legal_paths.append(path)
                legal_preds.append(legal_paths)
        else:
            legal_preds = batch_preds

        return legal_preds


# class LightweightViT(VisionTransformer):
#     def __init__(self, img_size=(14, 14), patch_size=1, embed_dim=64, depth=2, num_heads=2, mlp_ratio=2):
#         super().__init__(
#             img_size=img_size,
#             patch_size=patch_size,
#             embed_dim=embed_dim,
#             depth=depth,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=True,
#         )
#         # Override the patch embeddings to match the input size
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim
#         )


#     def forward_features(self, x):
#         # Flatten the patches
#         x = self.patch_embed(x)
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # Stolen cls token applied to all batches
#         x = torch.cat((cls_token, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         x = self.norm(x)
#         return x[:, 0]

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x


class SchemaGuidedInstructBertModelWithPromptReplace_Concatenate_GNN(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        vocab_size: int = None,
        use_rope: bool = True,
        biaffine_size: int = 512,
        label_mask_id: int = 4,
        text_mask_id: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        self.plm = AutoModel.from_pretrained(plm_dir)
        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        self.fusion = TrainableFusion(fusion=0.5)
        self.linear_proj = LinearWithAct(n_in=self.plm.config.hidden_size*2,
                                         n_out=self.plm.config.hidden_size,
                                         dropout=0.3)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        # _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        # mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # # bit_mask: (batch_size, 1, seq_len, seq_len)
        # bit_mask = (
        #     torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        # )
        bit_mask = (
            mask.gt(0).unsqueeze(1).unsqueeze(1).expand(bs, 1, seq_len, seq_len).float()
        )

        return bit_mask

    def prmopt_encoding(self, prompts_to_indices, hidden, prompts_ids, prompts_mask) -> torch.Tensor:
        # 1. Get the hidden_sentence_state from the last layer
        hidden_prmopt = self.input_encoding(prompts_ids, prompts_mask)
        hiddens = hidden_prmopt[:, -2, :]  # remove [sep] and .
        
        # Create a copy of hidden for updates
        updated_hidden = hidden.clone()

        # Flatten the indices for batch and prompt indexing
        batch_indices, prompt_indices = prompts_to_indices[:, :, -1].nonzero(as_tuple=True)

        # Select the corresponding prompt hiddens using advanced indexing
        selected_prompt_hiddens = hiddens[prompt_indices]

        # Concatenate hidden and prompt hidden states
        temp = torch.cat((hidden[batch_indices, prompt_indices], selected_prompt_hiddens), dim=1)

        # Apply linear projection
        prompt_hiddens_projected = self.linear_proj(temp)

        # Fusion operation
        updated_hidden[batch_indices, prompt_indices] = self.fusion(prompt_hiddens_projected, hidden[batch_indices, prompt_indices])

        return updated_hidden
    
    def forward(
        self, input_ids,prompts_to_indices,prompts_ids,mask,prompts_mask, labels=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        hidden = self.prmopt_encoding(prompts_to_indices,hidden,prompts_ids[0], prompts_mask[0])
        # (bs, 3, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        # edge_index = self.build_edge_index(logits,bs,seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss
            # batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            # results["pred"] = batch_positions

        if is_eval:
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        # legal_num_parts: tuple = (1, 2, 3),
        legal_num_parts: tuple = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # B x 3 x L x L
        if labels is None:
            # `labels` is used for upper bound analysis
            probs = logits.sigmoid()
            pred = (probs > top_p).long()
        else:
            pred = labels
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        # for pred, gold in zip(preds, kwargs.get("spans")):
        #     sorted_pred = sorted(set(tuple(x) for x in pred))
        #     sorted_gold = sorted(set(tuple(x) for x in gold))
        #     if sorted_pred != sorted_gold:
        #         breakpoint()

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        if legal_num_parts is not None:
            legal_preds = []
            for ins_paths in batch_preds:
                legal_paths = []
                for path in ins_paths:
                    if len(path) in legal_num_parts:
                        legal_paths.append(path)
                legal_preds.append(legal_paths)
        else:
            legal_preds = batch_preds

        return legal_preds


class SchemaGuidedInstructBertModelWithPromptReplace_Concatenate_PN(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        vocab_size: int = None,
        use_rope: bool = True,
        biaffine_size: int = 512,
        label_mask_id: int = 4,
        text_mask_id: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        self.plm = AutoModel.from_pretrained(plm_dir)
        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        self.fusion = TrainableFusion(fusion=0.5)
        self.linear_proj = LinearWithAct(n_in=self.plm.config.hidden_size*2,
                                         n_out=self.plm.config.hidden_size,
                                         dropout=0.3)
        
        self.fusion_neg = TrainableFusion(fusion=0.5)
        self.linear_proj_neg = LinearWithAct(n_in=self.plm.config.hidden_size*2,
                                         n_out=self.plm.config.hidden_size,
                                         dropout=0.3)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        # _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        # mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # # bit_mask: (batch_size, 1, seq_len, seq_len)
        # bit_mask = (
        #     torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        # )
        bit_mask = (
            mask.gt(0).unsqueeze(1).unsqueeze(1).expand(bs, 1, seq_len, seq_len).float()
        )

        return bit_mask

    def prmopt_encoding(self, prompts_to_indices, hidden, prompts_ids, prompts_mask) -> torch.Tensor:
        # 1. Get the hidden_sentence_state from the last layer
        hidden_prmopt = self.input_encoding(prompts_ids, prompts_mask)
        hiddens = hidden_prmopt[:, -2, :]  # remove [sep] and .
        
        # Create a copy of hidden for updates
        updated_hidden = hidden.clone()

        # Flatten the indices for batch and prompt indexing
        batch_indices, prompt_indices = prompts_to_indices[:, :, -1].nonzero(as_tuple=True)

        # Select the corresponding prompt hiddens using advanced indexing
        selected_prompt_hiddens = hiddens[prompt_indices]

        # Concatenate hidden and prompt hidden states
        temp = torch.cat((hidden[batch_indices, prompt_indices], selected_prompt_hiddens), dim=1)

        # Apply linear projection
        prompt_hiddens_projected = self.linear_proj(temp)

        # Fusion operation
        updated_hidden[batch_indices, prompt_indices] = self.fusion(prompt_hiddens_projected, hidden[batch_indices, prompt_indices])

        return updated_hidden
        
    def forward(
        self, input_ids,prompts_to_indices,prompts_ids,mask,prompts_mask, labels=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        hidden = self.prmopt_encoding(prompts_to_indices,hidden,prompts_ids[0], prompts_mask[0])
        # (bs, 3, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        # edge_index = self.build_edge_index(logits,bs,seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss
            # batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            # results["pred"] = batch_positions

        if is_eval:
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        # legal_num_parts: tuple = (1, 2, 3),
        legal_num_parts: tuple = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # B x 3 x L x L
        if labels is None:
            # `labels` is used for upper bound analysis
            probs = logits.sigmoid()
            pred = (probs > top_p).long()
        else:
            pred = labels
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        # for pred, gold in zip(preds, kwargs.get("spans")):
        #     sorted_pred = sorted(set(tuple(x) for x in pred))
        #     sorted_gold = sorted(set(tuple(x) for x in gold))
        #     if sorted_pred != sorted_gold:
        #         breakpoint()

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        if legal_num_parts is not None:
            legal_preds = []
            for ins_paths in batch_preds:
                legal_paths = []
                for path in ins_paths:
                    if len(path) in legal_num_parts:
                        legal_paths.append(path)
                legal_preds.append(legal_paths)
        else:
            legal_preds = batch_preds

        return legal_preds

class SchemaGuidedInstructBertModelWithPromptReplace_Event(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        vocab_size: int = None,
        use_rope: bool = True,
        biaffine_size: int = 512,
        label_mask_id: int = 4,
        text_mask_id: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        self.plm = AutoModel.from_pretrained(plm_dir)
        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        self.fusion = TrainableFusion(fusion=0.5)
        self.linear_proj = LinearWithAct(n_in=self.plm.config.hidden_size*2,
                                         n_out=self.plm.config.hidden_size,
                                         dropout=0.3)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        # _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        # mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # # bit_mask: (batch_size, 1, seq_len, seq_len)
        # bit_mask = (
        #     torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        # )
        bit_mask = (
            mask.gt(0).unsqueeze(1).unsqueeze(1).expand(bs, 1, seq_len, seq_len).float()
        )

        return bit_mask

    def prmopt_encoding(self, prompts_to_indices, hidden, prompts_ids, prompts_mask) -> torch.Tensor:
        # 1. Get the hidden_sentence_state from the last layer
        hidden_prmopt = self.input_encoding(prompts_ids, prompts_mask)
        hiddens = hidden_prmopt[:, -2, :]  # remove [sep] and .
        
        # Create a copy of hidden for updates
        updated_hidden = hidden.clone()

        # Flatten the indices for batch and prompt indexing
        batch_indices, prompt_indices = prompts_to_indices[:, :, -1].nonzero(as_tuple=True)

        # Select the corresponding prompt hiddens using advanced indexing
        selected_prompt_hiddens = hiddens[prompt_indices]

        # Concatenate hidden and prompt hidden states
        temp = torch.cat((hidden[batch_indices, prompt_indices], selected_prompt_hiddens), dim=1)
        temp = relevant_hidden = slerp_for_batch_hidden(relevant_hidden,temp,torch.full((relevant_hidden.shape[0],), 0.8).to(relevant_hidden.device))
        # Apply linear projection
        prompt_hiddens_projected = self.linear_proj(temp)

        # Fusion operation
        updated_hidden[batch_indices, prompt_indices] = temp

        return updated_hidden
    
    def get_prompt_hidden(self,prompts_ids, prompts_mask) ->torch.Tensor:
        hidden_prmopt = self.input_encoding(prompts_ids,prompts_mask)
        flipped_mask = torch.flip(prompts_mask,[1])
        last_valid_index = prompts_mask.size(1) - 1 - torch.argmax(flipped_mask,dim = 1)
        last_valid_index = (last_valid_index-1).unsqueeze(1).unsqueeze(2).expand(-1,-1,hidden_prmopt.size(2))
        hiddens = hidden_prmopt.gather(1,last_valid_index).squeeze(1)
        # hiddens = hidden_prmopt[:,-2,:].squeeze(1) # remove [sep] and .
        hiddens = hiddens.clone()
        return hiddens
    
    def update_with_prompt_hiddens(self, hidden, prompt_hiddens, prompts_to_indices):
        # 创建hidden的副本
        updated_hidden = hidden.clone()

        # 扁平化prompts_to_indices并提取所有相关的hidden states和prompt hiddens
        batch_indices, hidden_indices = torch.where(prompts_to_indices[..., -1] >= 0)
        relevant_hidden = updated_hidden[batch_indices, hidden_indices]
        relevant_prompts = prompt_hiddens[hidden_indices]

        # 批量操作
        concatenated = torch.cat([relevant_hidden, relevant_prompts], dim=1)
        projected = self.linear_proj(concatenated)
        fused = self.fusion(projected, relevant_hidden)

        # 更新updated_hidden
        updated_hidden[batch_indices, hidden_indices] = fused.reshape(-1, 1024)

        return updated_hidden

        
    def forward(
        self, input_ids,spans,prompts_to_indices,mask, labels=None,hidden_prmopt=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        hidden = self.update_with_prompt_hiddens(hidden,hidden_prmopt,prompts_to_indices)
        # (bs, 3, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        # edge_index = self.build_edge_index(logits,bs,seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss

            if is_eval:
                batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
                results["pred"] = batch_positions

        # if is_eval:
        #     # pdb.set_trace()
        #     droplabels = torch.zeros((bs, 1, seq_len, seq_len)).to(logits)
        #     for i in range(bs):
        #         for span in spans[i]:
        #             stolist = []
        #             for t in span:
        #                 if len(t)==1: stolist.append(t[0])
        #                 else: 
        #                     stolist.extend([t[0],t[1]])
        #             for m in stolist:
        #                 for n in stolist:
        #                     droplabels[i, 0, m, n] = 1
        #     droplabels = droplabels.repeat(1, cls_num, 1, 1)
        #     zeromask = droplabels != 0
        #     droplabels = droplabels * zeromask
        #     logits = logits * droplabels
        #     batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, spans=spans, **kwargs)
        #     results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        # legal_num_parts: tuple = (1, 2, 3),
        legal_num_parts: tuple = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # B x 3 x L x L
        if labels is None:
            # `labels` is used for upper bound analysis
            probs = logits.sigmoid()
            pred = (probs > top_p).long()
        else:
            pred = labels
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        # for pred, gold in zip(preds, kwargs.get("spans")):
        #     sorted_pred = sorted(set(tuple(x) for x in pred))
        #     sorted_gold = sorted(set(tuple(x) for x in gold))
        #     if sorted_pred != sorted_gold:
        #         breakpoint()

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        if legal_num_parts is not None:
            legal_preds = []
            for ins_paths in batch_preds:
                legal_paths = []
                for path in ins_paths:
                    if len(path) in legal_num_parts:
                        legal_paths.append(path)
                legal_preds.append(legal_paths)
        else:
            legal_preds = batch_preds

        return legal_preds



class SchemaGuidedInstructBertModelWithPromptReplace_Event_Layer_Level(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        vocab_size = 128100,
        use_rope: bool = True,
        biaffine_size: int = 512,
        label_mask_id: int = 4,
        text_mask_id: int = 7,
        dropout: float = 0.3,
        output_attentions = False
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope
        self.output_attentions = output_attentions
        # self.plm = Prompt_Fusion_Model.from_pretrained(plm_dir,strict = False)
        # self.plm = AutoModel.from_pretrained(plm_dir)
        self.plm = Prompt_Fusion_Model.from_local_pretrained(Prompt_Fusion_Model,plm_dir) 
        # # self.prompt_plm = AutoModel.from_pretrained(plm_dir)

        # self.prompt_plm = AutoModel.from_pretrained(plm_dir)
        # self.num_prompt_layer = 12
        # self._initialize_prompt_plm(plm_dir)

        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
            # todo:1.init with the number of 0.5*(num_layer) of self.plm
            # todo:2.init with the number of 1*(num_layer) of self.plm
            # todo:3.如果权重路径有prompt_plm的参数那么直接加载，没有则取self.plm的前12层
        # self.fusion = TrainableFusion(fusion=0.5)
        # self.linear_proj = LinearWithAct(n_in=self.plm.config.hidden_size*2,
        #                                  n_out=self.plm.config.hidden_size,
        #                                  dropout=0.3)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )


    # def _initialize_prompt_plm(self, plm_dir):
    #     # 从 plm_dir 加载检查点文件
    #     checkpoint = torch.load(plm_dir)
    #     prompt_weights = {k: v for k, v in checkpoint.items() if 'prompt_plm' in k}

    #     # 获取前 12 层的权重
    #     old_layers = self.plm.encoder.layer[:self.num_prompt_layer]
    #     new_config = AutoConfig.from_pretrained(plm_dir)
    #     new_config.num_hidden_layers = self.num_prompt_layer
    #     self.prompt_plm = AutoModel.from_config(new_config)
        
    #     # 复制前 12 层的权重
    #     for i in range(self.num_prompt_layer):
    #         self.prompt_plm.encoder.layer[i].load_state_dict(old_layers[i].state_dict())

    #     # 加载 prompt_plm 的权重
    #     self.prompt_plm.load_state_dict(prompt_weights, strict=False)

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.prompt_plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states = True,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        # _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        # mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # # bit_mask: (batch_size, 1, seq_len, seq_len)
        # bit_mask = (
        #     torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        # )
        bit_mask = (
            mask.gt(0).unsqueeze(1).unsqueeze(1).expand(bs, 1, seq_len, seq_len).float()
        )

        return bit_mask

    def prmopt_encoding(self, prompts_to_indices, hidden, prompts_ids, prompts_mask) -> torch.Tensor:
        # 1. Get the hidden_sentence_state from the last layer
        hidden_prmopt = self.input_encoding(prompts_ids, prompts_mask)
        hiddens = hidden_prmopt[:, -2, :]  # remove [sep] and .
        
        # Create a copy of hidden for updates
        updated_hidden = hidden.clone()

        # Flatten the indices for batch and prompt indexing
        batch_indices, prompt_indices = prompts_to_indices[:, :, -1].nonzero(as_tuple=True)

        # Select the corresponding prompt hiddens using advanced indexing
        selected_prompt_hiddens = hiddens[prompt_indices]

        # Concatenate hidden and prompt hidden states
        temp = torch.cat((hidden[batch_indices, prompt_indices], selected_prompt_hiddens), dim=1)

        # Apply linear projection
        prompt_hiddens_projected = self.linear_proj(temp)

        # Fusion operation
        updated_hidden[batch_indices, prompt_indices] = self.fusion(prompt_hiddens_projected, hidden[batch_indices, prompt_indices])

        return updated_hidden

    def input_encoding_prompt(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def get_prompt_hidden(self,prompts_ids, prompts_mask) ->torch.Tensor:
        hidden_prmopt = self.input_encoding(prompts_ids,prompts_mask)
        flipped_mask = torch.flip(prompts_mask,[1])
        last_valid_index = prompts_mask.size(1) - 1 - torch.argmax(flipped_mask,dim = 1)
        last_valid_index = (last_valid_index-1).unsqueeze(1).unsqueeze(2).expand(-1,-1,hidden_prmopt.size(2))
        hiddens = hidden_prmopt.gather(1,last_valid_index).squeeze(1)
        # hiddens = hidden_prmopt[:,-2,:].squeeze(1) # remove [sep] and .
        hiddens = hiddens.clone()
        return hiddens
    
    def update_with_prompt_hiddens(self, hidden, prompt_hiddens, prompts_to_indices):
        # 创建hidden的副本
        updated_hidden = hidden.clone()

        # 扁平化prompts_to_indices并提取所有相关的hidden states和prompt hiddens
        batch_indices, hidden_indices = torch.where(prompts_to_indices[..., -1] >= 0)
        relevant_hidden = updated_hidden[batch_indices, hidden_indices]
        relevant_prompts = prompt_hiddens[hidden_indices]

        # 批量操作
        concatenated = torch.cat([relevant_hidden, relevant_prompts], dim=1)
        projected = self.linear_proj(concatenated)
        fused = self.fusion(projected, relevant_hidden)

        # 更新updated_hidden
        updated_hidden[batch_indices, hidden_indices] = fused.reshape(-1, 1024)

        return updated_hidden

        
    # def forward(
    #     self, input_ids,spans,prompts_to_indices,mask, labels=None,hidden_prmopt=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    # ):
    #     bit_mask = self.build_bit_mask(mask)
    #     hidden = self.input_encoding(input_ids, mask)
    #     hidden = self.update_with_prompt_hiddens(hidden,hidden_prmopt,prompts_to_indices)
    #     # (bs, 3, seq_len, seq_len)
    #     logits = self.pointer(hidden)
    #     logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
    #     logits = logits / (self.biaffine_size**0.5)
    #     # # (bs, 3, seq_len, seq_len)
    #     bs, cls_num, seq_len, seq_len = logits.shape
    #     # edge_index = self.build_edge_index(logits,bs,seq_len)

    #     results = {"logits": logits}
    #     if labels is not None:
    #         loss = multilabel_categorical_crossentropy(
    #             logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
    #         )
    #         loss = loss.mean()
    #         results["loss"] = loss

    #         if is_eval:
    #             batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
    #             results["pred"] = batch_positions

    #     # if is_eval:
    #     #     # pdb.set_trace()
    #     #     droplabels = torch.zeros((bs, 1, seq_len, seq_len)).to(logits)
    #     #     for i in range(bs):
    #     #         for span in spans[i]:
    #     #             stolist = []
    #     #             for t in span:
    #     #                 if len(t)==1: stolist.append(t[0])
    #     #                 else: 
    #     #                     stolist.extend([t[0],t[1]])
    #     #             for m in stolist:
    #     #                 for n in stolist:
    #     #                     droplabels[i, 0, m, n] = 1
    #     #     droplabels = droplabels.repeat(1, cls_num, 1, 1)
    #     #     zeromask = droplabels != 0
    #     #     droplabels = droplabels * zeromask
    #     #     logits = logits * droplabels
    #     #     batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, spans=spans, **kwargs)
    #     #     results["pred"] = batch_positions
    #     return results

    def forward(
        self, input_ids,prompts_ids,prompts_mask,spans,prompts_to_indices,mask, labels=None, 
        is_eval=False, top_p=0.5, top_k=-1, output_all_encoded_layers=True,
        **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        # hidden = self.input_encoding(input_ids, mask)
        # hidden = self.update_with_prompt_hiddens(hidden,hidden_prmopt,prompts_to_indices)
        # # (bs, 3, seq_len, seq_len)
        result = self.plm(input_ids = input_ids,
            attention_mask = mask,
            prompts_ids = prompts_ids[0],
            prompts_mask = prompts_mask[0],
            prompts_to_indices = prompts_to_indices,
            )
        # input_embeddings = self.plm.embeddings(input_ids = input_ids,mask = mask)
        # prompt_input_embeddings = self.plm.prompt_embeddings(input_ids = prompts_ids[0],mask = prompts_mask[0])
        # hidden_states = input_embeddings
        # prompt_hidden_states = prompt_input_embeddings
        # all_encoder_layers = []
        # all_attentions = []
        # for i,layer_module in enumerate(self.plm.encoder.layer):
        #     hidden_states = layer_module(hidden_states,mask)
        #     if i < self.num_prompt_layer:
        #         prompt_hidden_states = self.prompt_plm.encoder.layer[i](prompt_hidden_states,prompts_mask)
        #         if self.output_attentions:
        #             prompt_attentions,prompt_hidden_states = prompt_hidden_states
        #             all_attentions.append(prompt_attentions)
        #     if self.output_attentions:
        #         attentions, hidden_states = hidden_states
        #         all_attentions.append(attentions)
        #         hidden_states = self.update_with_prompt_hiddens(hidden_states,prompt_hidden_states,prompts_to_indices)
        #     else:
        #         hidden_states = self.update_with_prompt_hiddens(hidden_states,prompt_hidden_states,prompts_to_indices)
        #     if output_all_encoded_layers:
        #         all_encoder_layers.append(hidden_states)
        # if not all_encoder_layers:
        #     all_encoder_layers.append(hidden_states)
        hidden_states = result['last_hidden_state']
        logits = self.pointer(hidden_states)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        # edge_index = self.build_edge_index(logits,bs,seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss

            if is_eval:
                batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
                results["pred"] = batch_positions

        # if is_eval:
        #     # pdb.set_trace()
        #     droplabels = torch.zeros((bs, 1, seq_len, seq_len)).to(logits)
        #     for i in range(bs):
        #         for span in spans[i]:
        #             stolist = []
        #             for t in span:
        #                 if len(t)==1: stolist.append(t[0])
        #                 else: 
        #                     stolist.extend([t[0],t[1]])
        #             for m in stolist:
        #                 for n in stolist:
        #                     droplabels[i, 0, m, n] = 1
        #     droplabels = droplabels.repeat(1, cls_num, 1, 1)
        #     zeromask = droplabels != 0
        #     droplabels = droplabels * zeromask
        #     logits = logits * droplabels
        #     batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, spans=spans, **kwargs)
        #     results["pred"] = batch_positions
        if self.output_attentions:
            results['output_attentions'] = all_attentions,all_encoder_layers
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        # legal_num_parts: tuple = (1, 2, 3),
        legal_num_parts: tuple = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # B x 3 x L x L
        if labels is None:
            # `labels` is used for upper bound analysis
            probs = logits.sigmoid()
            pred = (probs > top_p).long()
        else:
            pred = labels
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        # for pred, gold in zip(preds, kwargs.get("spans")):
        #     sorted_pred = sorted(set(tuple(x) for x in pred))
        #     sorted_gold = sorted(set(tuple(x) for x in gold))
        #     if sorted_pred != sorted_gold:
        #         breakpoint()

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        if legal_num_parts is not None:
            legal_preds = []
            for ins_paths in batch_preds:
                legal_paths = []
                for path in ins_paths:
                    if len(path) in legal_num_parts:
                        legal_paths.append(path)
                legal_preds.append(legal_paths)
        else:
            legal_preds = batch_preds

        return legal_preds



# 3.Attention Fusion 
class AttentionFusionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionFusionLayer, self).__init__()
        self.query = nn.Linear(input_size, output_size)
        self.key = nn.Linear(input_size, output_size)
        self.value = nn.Linear(input_size, output_size)

    def forward(self, input1, input2, input3):
        # 生成查询、键、值
        q = self.query(input1)
        k = self.key(input2)
        v = self.value(input3)

        # 计算注意力权重
        attn_weights = torch.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2)), dim=-1)
        output = torch.bmm(attn_weights, v.unsqueeze(1)).squeeze(1)
        return output


import math

class ImprovedAttentionFusionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImprovedAttentionFusionLayer, self).__init__()
        self.projrct1 = nn.Linear(input_size, output_size)
        self.projrct2 = nn.Linear(input_size, output_size)
        self.projrct3 = nn.Linear(input_size, output_size)
        self.query_transform = nn.Linear(input_size, output_size)
        self.key_transform = nn.Linear(input_size, output_size)
        self.value_transform = nn.Linear(input_size, output_size)
        self.scale = 1 / math.sqrt(output_size)

    def forward(self, input1, input2, input3):
        # 保持input1的原始信息并融合input2和input3
        input1 = self.projrct1(input1)
        input2 = self.projrct2(input2)
        input3 = self.projrct3(input3)
        q = self.query_transform(input1 + input2)  # 使用input1和input2的和
        k = self.key_transform(input1 + input3)  # 使用input1和transformed_input3的和
        v = self.value_transform(input1)

        # 计算注意力权重，添加缩放因子
        attn_weights = torch.softmax(self.scale * torch.bmm(q.unsqueeze(1), k.unsqueeze(2)), dim=-1)
        output = torch.bmm(attn_weights, v.unsqueeze(1)).squeeze(1)
        return output

class GatedFusionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(GatedFusionLayer, self).__init__()
        self.transform = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(3 * output_size, output_size)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2, input3):
        # Transform inputs
        transformed_input3 = self.transform(input3)

        # Concatenate inputs
        concatenated_inputs = torch.cat([input1,input2, transformed_input3], dim=1)

        # Compute gating values
        gating_values = self.sigmoid(self.gate(concatenated_inputs))

        # Apply gate to the original input
        gated_input = gating_values * input1

        # Apply activation
        output = self.activation(gated_input)

        return output
    
    
class LinearFusionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearFusionLayer, self).__init__()
        self.transform1 = nn.Linear(input_size*2, output_size)
        self.transform2 = nn.Linear(input_size*2, output_size)
        self.combine = nn.Linear(input_size*2, output_size)


    def forward(self, input1, input2, input3):
        # Transform inputs
        pos = torch.cat([input1,input2],dim=1)
        pos = self.transform1(pos)
        neg = torch.cat([input1,input3],dim=1)
        neg = self.transform2(neg)
        combined = torch.cat([pos, neg], dim=1)
        output = self.combine(combined)

        return output


class SchemaGuidedInstructBertModelWithPromptReplace_PN_Event(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        vocab_size: int = None,
        use_rope: bool = True,
        biaffine_size: int = 512,
        label_mask_id: int = 4,
        text_mask_id: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        self.plm = AutoModel.from_pretrained(plm_dir)
        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        self.fusion = LinearFusionLayer(input_size=self.plm.config.hidden_size,
                                        output_size=self.plm.config.hidden_size)
        # self.linear_proj = LinearWithAct(n_in=self.plm.config.hidden_size*3,
        #                                 n_out=self.plm.config.hidden_size,
        #                                 dropout=0.3)
        self.fusion_res = TrainableFusion(fusion=0.5)
        self.hidden_size = self.plm.config.hidden_size
        
        
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        # _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        # mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # # bit_mask: (batch_size, 1, seq_len, seq_len)
        # bit_mask = (
        #     torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        # )
        bit_mask = (
            mask.gt(0).unsqueeze(1).unsqueeze(1).expand(bs, 1, seq_len, seq_len).float()
        )

        return bit_mask

    def prmopt_encoding(self, prompts_to_indices, hidden, prompts_ids, prompts_mask) -> torch.Tensor:
        # 1. Get the hidden_sentence_state from the last layer
        hidden_prmopt = self.input_encoding(prompts_ids, prompts_mask)
        hiddens = hidden_prmopt[:, -2, :]  # remove [sep] and .
        
        # Create a copy of hidden for updates
        updated_hidden = hidden.clone()

        # Flatten the indices for batch and prompt indexing
        batch_indices, prompt_indices = prompts_to_indices[:, :, -1].nonzero(as_tuple=True)

        # Select the corresponding prompt hiddens using advanced indexing
        selected_prompt_hiddens = hiddens[prompt_indices]

        # Concatenate hidden and prompt hidden states
        temp = torch.cat((hidden[batch_indices, prompt_indices], selected_prompt_hiddens), dim=1)

        # Apply linear projection
        prompt_hiddens_projected = self.linear_proj(temp)

        # Fusion operation
        updated_hidden[batch_indices, prompt_indices] = self.fusion(prompt_hiddens_projected, hidden[batch_indices, prompt_indices])

        return updated_hidden
    
    def get_prompt_hidden(self,prompts_ids, prompts_mask) ->torch.Tensor:
        hidden_prmopt = self.input_encoding(prompts_ids,prompts_mask)
        flipped_mask = torch.flip(prompts_mask,[1])
        last_valid_index = prompts_mask.size(1) - 1 - torch.argmax(flipped_mask,dim = 1)
        last_valid_index = (last_valid_index-1).unsqueeze(1).unsqueeze(2).expand(-1,-1,hidden_prmopt.size(2))
        hiddens = hidden_prmopt.gather(1,last_valid_index).squeeze(1)
        # hiddens = hidden_prmopt[:,-2,:].squeeze(1) # remove [sep] and .
        hiddens = hiddens.clone()
        return hiddens
    
    def update_with_prompt_hiddens(self, hidden, prompt_hiddens, prompt_hiddens_neg,prompts_to_indices,prompts_to_indices_neg):
        # 创建hidden的副本
        updated_hidden = hidden.clone()

        # 扁平化prompts_to_indices并提取所有相关的hidden states和prompt hiddens
        batch_indices, hidden_indices = torch.where(prompts_to_indices[..., -1] >= 0)
        relevant_hidden = updated_hidden[batch_indices, hidden_indices]
        relevant_prompts = prompt_hiddens[hidden_indices]
        
        # 扁平化prompts_to_indices_neg并提取所有相关的hidden states和prompt hiddens
        batch_indices, hidden_indices_neg = torch.where(prompts_to_indices_neg[..., -1] >= 0)
        # relevant_hidden = updated_hidden[batch_indices, hidden_indices_neg]
        relevant_prompts_neg = prompt_hiddens_neg[hidden_indices_neg]

        # 批量操作
        # concatenated = torch.cat([relevant_hidden, relevant_prompts,relevant_prompts_neg], dim=1)
        # projected = self.linear_proj(concatenated)
        # fused = self.fusion(projected, relevant_hidden)
        fused = self.fusion(relevant_hidden, relevant_prompts,relevant_prompts_neg)
        # updated_hidden[batch_indices, hidden_indices] = self.fusion_res(fused.reshape(-1, 1024),updated_hidden[batch_indices, hidden_indices])
        # 更新updated_hidden
        updated_hidden[batch_indices, hidden_indices] = fused.reshape(-1, 1024)

        return updated_hidden
            
    def forward(
        self, input_ids,spans,prompts_to_indices,prompts_to_indices_neg,mask, labels=None,hidden_prmopt=None, hidden_prmopt_neg= None,is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        hidden = self.update_with_prompt_hiddens(hidden,hidden_prmopt,hidden_prmopt_neg,prompts_to_indices,prompts_to_indices_neg)        # (bs, 3, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        # edge_index = self.build_edge_index(logits,bs,seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss
            if is_eval:
                batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
                results["pred"] = batch_positions
        # if is_eval:
        #     # pdb.set_trace()
        #     droplabels = torch.zeros((bs, 1, seq_len, seq_len)).to(logits)
        #     for i in range(bs):
        #         for span in spans[i]:
        #             stolist = []
        #             for t in span:
        #                 if len(t)==1: stolist.append(t[0])
        #                 else: 
        #                     stolist.extend([t[0],t[1]])
        #             for m in stolist:
        #                 for n in stolist:
        #                     droplabels[i, 0, m, n] = 1
        #     droplabels = droplabels.repeat(1, cls_num, 1, 1)
        #     zeromask = droplabels != 0
        #     droplabels = droplabels * zeromask
        #     logits = logits * droplabels
        #     batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, spans=spans, **kwargs)
        #     results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        # legal_num_parts: tuple = (1, 2, 3),
        legal_num_parts: tuple = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # B x 3 x L x L
        if labels is None:
            # `labels` is used for upper bound analysis
            probs = logits.sigmoid()
            pred = (probs > top_p).long()
        else:
            pred = labels
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        # for pred, gold in zip(preds, kwargs.get("spans")):
        #     sorted_pred = sorted(set(tuple(x) for x in pred))
        #     sorted_gold = sorted(set(tuple(x) for x in gold))
        #     if sorted_pred != sorted_gold:
        #         breakpoint()

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        if legal_num_parts is not None:
            legal_preds = []
            for ins_paths in batch_preds:
                legal_paths = []
                for path in ins_paths:
                    if len(path) in legal_num_parts:
                        legal_paths.append(path)
                legal_preds.append(legal_paths)
        else:
            legal_preds = batch_preds

        return legal_preds