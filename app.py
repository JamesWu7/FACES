import html
import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from faces_inference import predict_faces


APP_ROOT = Path(__file__).resolve().parent

CLINICAL_NOTE = (
    "本工具用于 FACES 研究框架下的科研展示和临床分诊辅助。模型输出表示相对支持度，"
    "不能作为独立诊断结论，也不能替代 X-ray、MRI、遗传检测、病史查体和脊柱外科/"
    "医学遗传专科评估。若结果提示 scoliosis 或 SS 相关支持度较高，建议进入标准临床评估流程。"
)


def _to_dataframe(items, label_col="类别"):
    return pd.DataFrame(
        [
            {
                label_col: item["class_name"],
                "概率": round(item["probability"], 3),
                "百分比": f"{item['probability_percent']:.1f}%",
            }
            for item in items
        ]
    )


def _plot_probabilities(items, title, limit=None):
    plot_items = items[:limit] if limit else items
    labels = [item["class_name"] for item in plot_items][::-1]
    values = [item["probability"] for item in plot_items][::-1]

    fig_height = max(3.2, 0.38 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    bars = ax.barh(range(len(labels)), values, color="#2A7F62", edgecolor="#1D4E40")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Model-derived support")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            min(value + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left" if value < 0.92 else "right",
            fontsize=9,
        )
    fig.tight_layout()
    return fig


def _summary_markdown(result):
    top = result["top_summary"]
    subtype_lines = "\n".join(
        [
            f"- {item['class_name']}: {item['probability']:.3f}"
            for item in top["ss_subtype_top3"]
        ]
    )
    warnings = "\n".join([f"- {html.escape(w)}" for w in result["warnings"]])
    warning_block = f"\n\n**提示**\n{warnings}" if warnings else ""

    return f"""
**分层预测摘要**

- 二分类筛查 Top-1: `{top["screening_result"]}` ({top["screening_probability"]:.3f})
- 病因大类 Top-1: `{top["etiology_top1"]}` ({top["etiology_probability"]:.3f})
- SS 细分类 Top-3:
{subtype_lines}

**运行信息**

- 人脸裁剪: `{result["face_crop_method"]}`
- 推理设备: `{result["device"]}`
{warning_block}
"""


def run_faces_app(image: Image.Image):
    try:
        result = predict_faces(image)
    except Exception as exc:
        message = f"无法完成推理：{exc}"
        empty = pd.DataFrame(columns=["类别", "概率", "百分比"])
        return None, message, empty, empty, empty, None, CLINICAL_NOTE

    binary_df = _to_dataframe(result["binary"])
    etiology_df = _to_dataframe(result["etiology"])
    subtype_df = _to_dataframe(result["ss_subtypes"])
    fig = _plot_probabilities(result["ss_subtypes"], "SS subtype ranked probabilities")

    return (
        result["cropped_face"],
        _summary_markdown(result),
        binary_df,
        etiology_df,
        subtype_df,
        fig,
        CLINICAL_NOTE,
    )


with gr.Blocks(title="FACES", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# FACES
**Facial Analysis for Classification of Etiology in Scoliosis**

上传一张清晰、无遮挡、正面人脸照片，系统将按 FACES 分层框架输出二分类筛查、AIS/CMS/SS 病因大类和 SS 11 类细分支持度。
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传正面人脸照片")
            submit_btn = gr.Button("开始分层预测", variant="primary")
        with gr.Column(scale=1):
            cropped_output = gr.Image(type="pil", label="人脸裁剪预览", interactive=False)
            summary_output = gr.Markdown(label="分层预测摘要")

    with gr.Tabs():
        with gr.Tab("二分类筛查"):
            binary_table = gr.Dataframe(label="HC vs scoliosis", interactive=False)
        with gr.Tab("病因大类"):
            etiology_table = gr.Dataframe(label="AIS / CMS / SS", interactive=False)
        with gr.Tab("SS 11类细分"):
            subtype_table = gr.Dataframe(label="SS subtype probabilities", interactive=False)
            subtype_plot = gr.Plot(label="SS subtype ranked probabilities")
        with gr.Tab("医学说明"):
            clinical_note = gr.Textbox(
                value=CLINICAL_NOTE,
                label="Clinical note",
                lines=5,
                interactive=False,
            )

    submit_btn.click(
        fn=run_faces_app,
        inputs=image_input,
        outputs=[
            cropped_output,
            summary_output,
            binary_table,
            etiology_table,
            subtype_table,
            subtype_plot,
            clinical_note,
        ],
    )


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port, show_api=False)
