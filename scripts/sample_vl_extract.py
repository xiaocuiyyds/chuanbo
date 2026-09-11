"""
一次性抽样脚本：挑几页含图片/表格的页面，用 Qwen-VL 转录成 Markdown，
人工对比效果和 token 消耗，再决定要不要把这条链路接进正式的 pdf_loader。
"""

import base64
import sys
from pathlib import Path

import fitz
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

import os  # noqa: E402

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VL_MODEL = "qwen-vl-plus"
ZOOM = 2.5  # ~180 DPI，兼顾清晰度和图片体积

PROMPT = (
    "请将这页船舶设备手册的内容完整转录为 Markdown。"
    "表格请转成 Markdown 表格；如果是接线图/流程图/示意图，"
    "请用文字描述图中的关键组件、标注和连接关系；"
    "保持原文语言，不要翻译，不要编造图中没有的内容。"
)

SAMPLES = {
    "KC-700 Manual.pdf": [3, 40, 80, 120, 160],
    "KONGSBERG AC600_MEB_FPP +(翻译结果).pdf": [5, 30, 60, 90],
}

OUT_DIR = Path(__file__).parent.parent / "ocr_samples"


def render_page(pdf_path: Path, page_no: int) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_no - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
        return pix.tobytes("png")
    finally:
        doc.close()


def main():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("DASHSCOPE_API_KEY 未设置，检查 .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
    OUT_DIR.mkdir(exist_ok=True)

    total_prompt_tokens = 0
    total_completion_tokens = 0

    for fname, pages in SAMPLES.items():
        pdf_path = Path(__file__).parent.parent / fname
        if not pdf_path.exists():
            print(f"跳过，找不到文件: {fname}")
            continue

        for page_no in pages:
            print(f"\n=== {fname} 第{page_no}页 ===")
            png_bytes = render_page(pdf_path, page_no)
            b64 = base64.b64encode(png_bytes).decode()

            img_path = OUT_DIR / f"{pdf_path.stem}_p{page_no}.png"
            img_path.write_bytes(png_bytes)

            response = client.chat.completions.create(
                model=VL_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
            )

            markdown = response.choices[0].message.content
            usage = response.usage
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

            md_path = OUT_DIR / f"{pdf_path.stem}_p{page_no}.md"
            md_path.write_text(markdown, encoding="utf-8")

            print(f"输入token={usage.prompt_tokens} 输出token={usage.completion_tokens}")
            print(markdown[:300])
            print("...(完整内容见", md_path.name, ")")

    print("\n=== 汇总 ===")
    print(f"总输入token: {total_prompt_tokens}")
    print(f"总输出token: {total_completion_tokens}")
    n_pages = sum(len(v) for v in SAMPLES.values())
    print(f"共测试 {n_pages} 页，平均每页输入 {total_prompt_tokens / n_pages:.0f} token，"
          f"输出 {total_completion_tokens / n_pages:.0f} token")
    print(f"结果保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()
