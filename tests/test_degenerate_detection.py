"""转录退化检测的回归测试。

这几个阈值是对着真实索引反复调出来的：要能抓到"同一句套话套到几十个位号上"的退化输出，
又不能把正常的 Markdown 表格误判成退化（表格行天然高度雷同）。调整阈值时必须让这些用例
继续通过，否则要么漏掉坏页，要么每次入库都白白多花一倍的转录调用。

样例取自真实索引中检出的退化页面。
"""

from rag.pdf_loader import _degenerate_reason


def test_truncated_output_is_flagged():
    assert _degenerate_reason("正常的一段转录内容。", "length") is not None


def test_normal_finish_is_not_flagged():
    assert _degenerate_reason("# 标题\n\n一段正常的手册内容，讲述主机滑油系统的工作原理。", "stop") is None


def test_short_text_is_never_flagged():
    """内容行太少时不做判断，避免误伤封面、章节页这类天然很短的页面。"""
    assert _degenerate_reason("# 第 3 章 消防系统\n\nDraft Manual for Review", "stop") is None


def test_same_sentence_repeated_across_tags_is_flagged():
    """真实退化形态：模型放弃读图，给每个位号套同一句废话。取自 Machinery p184。"""
    text = "## 管路图\n\n" + "\n".join(
        f"- **ASCO{i}**: Air Pressure Control Valve" for i in range(300, 360)
    )
    reason = _degenerate_reason(text, "stop")
    assert reason is not None
    assert "重复套话" in reason


def test_alternating_templates_are_flagged():
    """退化时模型常常几个模板轮流用，只看最高频的那一个会被摊薄漏掉。取自 Machinery p112。"""
    lines = []
    for i in range(60):
        lines.append(f"- **GFP{i}**: Located near the bottom of the diagram, connected to the drain path.")
        lines.append(f"- **GFP{i}**: Another component in the control air system, with notes on its function.")
        lines.append(f"- **GFP{i}**: Connected to the fuel gas valve unit, with annotations on its role.")
    assert _degenerate_reason("## 图\n\n" + "\n".join(lines), "stop") is not None


def test_indented_sub_bullets_are_flagged():
    """位号在父行、重复句在缩进子项上的形态。取自 Machinery p257。"""
    lines = []
    for i in range(40):
        lines.append(f"- **GSF{i}**")
        lines.append("  - Located near the main supply line.")
        lines.append("  - Connected to the main supply line via a vertical yellow line.")
    assert _degenerate_reason("\n".join(lines), "stop") is not None


def test_markdown_table_is_not_flagged():
    """正常转录的表格行高度雷同，绝不能被当成退化——否则每张表都要白白重试一次。"""
    header = "| Station | Status | Error | Spare Time | Net State |\n|---|---|---|---|---|"
    rows = [f"| PS{i:03d} | Operational | NOKNE | 0 | OK |" for i in range(100)]
    assert _degenerate_reason(header + "\n" + "\n".join(rows), "stop") is None


def test_repeated_page_footer_is_not_flagged():
    """页脚水印重复出现是正常的，不该触发重试。"""
    text = "# 第 4 章\n\n" + "\n\n".join(
        ["**Draft Manual for Review & Comment**", "正文段落，介绍消防总管的布置和压力维持方式。"] * 8
    )
    assert _degenerate_reason(text, "stop") is None


def test_normal_numbered_procedure_is_not_flagged():
    """编号步骤各不相同，不该被当成重复套话。"""
    steps = [
        "1. 打开滑油泵吸入阀 LOV-101。",
        "2. 启动备用滑油泵并确认压力建立。",
        "3. 检查滑油滤器压差报警是否复位。",
        "4. 确认主轴承入口压力高于 0.4 MPa。",
        "5. 记录运行参数并通知机舱值班。",
        "6. 在轮机日志中登记本次操作。",
        "7. 复查冷却器出口温度是否正常。",
        "8. 将泵组切换回自动待机模式。",
    ]
    assert _degenerate_reason("## 操作步骤\n\n" + "\n".join(steps), "stop") is None
