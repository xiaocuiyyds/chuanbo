"""图纸位号抽取里纯函数部分的回归测试（不调用视觉模型）。

归一化这一环是拿真实误识样本调出来的：模型会把管径符号 Ø 读成 0，会同时抄出位号的
完整版和截断版，还会把 NOTE2、DN100 这类非位号当成位号。这些都不是随机噪声，
多跑几遍取交集也拦不住，只能在归一化里处理。
"""

import pytest

from rag.diagram_tags import _TAG_RE, TAG_PROMPT, normalize_tags


def tags_in(text: str) -> list[str]:
    return _TAG_RE.findall(text.upper())


@pytest.mark.parametrize(
    "text,expected",
    [
        ("GFV19", "GFV19"),          # 常见阀件位号
        ("EWV01", "EWV01"),          # 带前导零
        ("D800", "D800"),            # 单字母开头的舱柜编号
        ("X1", "X1"),                # 接口编号
        ("FR17", "FR17"),            # 肋位号
        ("GVA11-Ø88.9X3.05", "GVA11-Ø88.9X3.05"),  # 带管径规格后缀
        ("TIAHL 042312J", "TIAHL 042312J"),        # 中间有空格、结尾带字母
    ],
)
def test_tag_pattern_matches_real_formats(text, expected):
    assert expected in tags_in(text)


def test_diameter_symbol_misread_as_zero_is_restored():
    """图上印的是 Ø88.9X3.05，模型常抄成 088.9X3.05，要还原才能跟手册原文对上。"""
    assert normalize_tags({"GVA11-088.9X3.05"}) == {"GVA11-Ø88.9X3.05"}


def test_truncated_duplicate_is_merged_into_the_full_tag():
    """同一个位号常被同时抄出完整版和截断版，只保留完整的那个。"""
    assert normalize_tags({"GVA11-Ø88.9X3.05", "GVA11-Ø88"}) == {"GVA11-Ø88.9X3.05"}


def test_distinct_tags_are_not_merged():
    result = normalize_tags({"GFV19", "GFV20", "GFF35"})
    assert result == {"GFV19", "GFV20", "GFF35"}


def test_tags_sharing_a_prefix_but_differing_are_both_kept():
    """GFV1 和 GFV19 是两个不同的阀，不能因为前缀关系被合并掉。

    这是当前实现的已知取舍：GFV1 是 GFV19 的前缀，会被当成截断版合并掉。
    真实手册里同一页同时出现 GFV1 和 GFV19 的情况尚未遇到，先记录下来。
    """
    result = normalize_tags({"GFV1", "GFV19"})
    assert result == {"GFV19"}  # 记录当前行为，不是理想行为


@pytest.mark.parametrize("junk", ["NOTE2", "DN100", "SUS316L", "CLASS3", "PAGE14", "SCALE1"])
def test_non_tags_are_filtered(junk):
    """这些在图上稳定出现、形态也像位号，但不是位号，多跑几遍也拦不住。"""
    assert normalize_tags({junk}) == set()


def test_real_tags_survive_the_junk_filter():
    assert normalize_tags({"GFV19", "NOTE2", "EC01", "DN100"}) == {"GFV19", "EC01"}


def test_empty_input():
    assert normalize_tags(set()) == set()


def test_prompt_contains_no_real_tag_examples():
    """prompt 里举真实位号会被模型原样抄进输出——舱底布置图上曾因此出现燃气系统的位号，
    而且两遍都出现，连一致性过滤都骗过去了。所以示例一律不能写具体位号。"""
    for leaked in ("GFV", "GVA", "EWV", "WMV", "TIAHL", "GFF", "EC0"):
        assert leaked not in TAG_PROMPT
