"""Interleave reasoning v6: v5 plus a draft answer before the decision point.

与 v5 的行格式相同，只多一种行内标记 ``<|noloss|>...<|/noloss|>``：包住的 token 不计 CE loss
（草答字母；草答对时深度分支里的门控 token ``<GENERATE_DEPTH>``）。

标记必须留在所在文本段内部处理——拆成单独的无 loss 文本段会多出 ``<|im_start|>/<|im_end|>``，
改变模型看到的 token 流。所以这里在分词前去掉标记、记下被包住的 token 在该段 ``text_ids`` 里的位置，
写进该段 sequence_plan 的 ``noloss_pos``，由 ``PackedDataset.pack_sequence`` 从 CE 里剔除。
标记是纯字符串，不注册成 token，模型永远看不到。
"""
import re

from .interleave_reasoning_dataset import InterleaveReasoningDatasetDepth

NOLOSS_START = "<|noloss|>"
NOLOSS_END = "<|/noloss|>"
_NOLOSS_PATTERN = re.compile(f"{re.escape(NOLOSS_START)}(.*?){re.escape(NOLOSS_END)}", flags=re.DOTALL)


def strip_noloss(text):
    """去掉标记。返回 (干净文本, [(字符起, 字符止), ...])，span 指向干净文本里被包住的内容。"""
    pieces, spans, cursor, length = [], [], 0, 0
    for m in _NOLOSS_PATTERN.finditer(text):
        pre = text[cursor:m.start()]
        pieces.append(pre)
        length += len(pre)
        body = m.group(1)
        spans.append((length, length + len(body)))
        pieces.append(body)
        length += len(body)
        cursor = m.end()
    pieces.append(text[cursor:])
    clean = "".join(pieces)
    assert NOLOSS_START not in clean and NOLOSS_END not in clean, f"标记不成对: {text[:200]!r}"
    return clean, spans


class InterleaveReasoningV6Dataset(InterleaveReasoningDatasetDepth):

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        clean, spans = strip_noloss(text)
        data = super()._add_text(data, clean, need_loss=need_loss, enable_cfg=enable_cfg)
        if spans and need_loss:
            data['sequence_plan'][-1]['noloss_pos'] = self._noloss_positions(
                clean, spans, data['text_ids_list'][-1])
        return data

    def _noloss_positions(self, text, spans, text_ids):
        """字符 span -> text_ids 下标。span 两端必须落在整段分词的 token 边界上，否则会切开一个 token。"""
        pos = []
        for start, end in spans:
            a = self.tokenizer.encode(text[:start])
            b = self.tokenizer.encode(text[:end])
            assert text_ids[:len(a)] == a and text_ids[:len(b)] == b, \
                f"no-loss span 不在 token 边界上: {text[start:end]!r}"
            assert len(b) > len(a), f"空的 no-loss span: {text[start:end]!r}"
            pos.extend(range(len(a), len(b)))
        return pos
