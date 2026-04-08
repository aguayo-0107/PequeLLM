from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required to generate the PDF presentation. "
        "Install it in your environment before running report generation."
    ) from exc


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _latest(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(values[-1])


def _load_train_metrics(run_dir: Path) -> Dict[str, np.ndarray]:
    rows = _read_csv(run_dir / "train_metrics.csv")
    if not rows:
        return {key: np.array([], dtype=np.float64) for key in ["iter", "train_loss", "val_loss", "lr", "global_grad_norm", "probe_gap", "probe_knn_purity"]}

    columns = {
        "iter": np.array([_to_float(r.get("iter", "")) for r in rows], dtype=np.float64),
        "train_loss": np.array([_to_float(r.get("train_loss", "")) for r in rows], dtype=np.float64),
        "val_loss": np.array([_to_float(r.get("val_loss", "")) for r in rows], dtype=np.float64),
        "lr": np.array([_to_float(r.get("lr", "")) for r in rows], dtype=np.float64),
        "global_grad_norm": np.array([_to_float(r.get("global_grad_norm", "")) for r in rows], dtype=np.float64),
        "probe_gap": np.array([_to_float(r.get("probe_gap", "")) for r in rows], dtype=np.float64),
        "probe_knn_purity": np.array([_to_float(r.get("probe_knn_purity", "")) for r in rows], dtype=np.float64),
    }
    return columns


def _load_grad_heatmap(run_dir: Path, top_k_layers: int = 16) -> Tuple[np.ndarray, List[str], List[int]]:
    rows = _read_csv(run_dir / "grad_norms_by_layer.csv")
    if not rows:
        return np.zeros((1, 1), dtype=np.float64), ["no_data"], [0]

    layer_values: Dict[str, List[float]] = {}
    for row in rows:
        layer = row.get("layer_name", "unknown")
        layer_values.setdefault(layer, []).append(_to_float(row.get("grad_norm", "nan")))

    ranked_layers = sorted(
        layer_values.keys(),
        key=lambda name: float(np.nanmean(np.array(layer_values[name], dtype=np.float64))),
        reverse=True,
    )
    keep_layers = ranked_layers[:top_k_layers]

    iter_set = sorted({int(_to_float(row.get("iter", "0"), default=0.0)) for row in rows})
    iter_to_idx = {it: idx for idx, it in enumerate(iter_set)}
    layer_to_idx = {name: idx for idx, name in enumerate(keep_layers)}

    matrix = np.full((len(iter_set), len(keep_layers)), np.nan, dtype=np.float64)
    for row in rows:
        layer = row.get("layer_name", "")
        if layer not in layer_to_idx:
            continue
        it = int(_to_float(row.get("iter", "0"), default=0.0))
        i_idx = iter_to_idx[it]
        l_idx = layer_to_idx[layer]
        matrix[i_idx, l_idx] = _to_float(row.get("grad_norm", "nan"))

    # Fill gaps with previous value along iterations to make heatmap readable.
    for col in range(matrix.shape[1]):
        col_vals = matrix[:, col]
        last_val = np.nan
        for row_idx in range(len(col_vals)):
            if math.isnan(col_vals[row_idx]):
                if not math.isnan(last_val):
                    col_vals[row_idx] = last_val
            else:
                last_val = col_vals[row_idx]
        matrix[:, col] = col_vals
    matrix = np.nan_to_num(matrix, nan=0.0)
    return matrix, keep_layers, iter_set


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_umap_images(run_dir: Path) -> List[Path]:
    img_dir = run_dir / "umap_snapshots"
    if not img_dir.exists():
        return []
    images = sorted(img_dir.glob("*.png"))
    if len(images) <= 3:
        return images
    return [images[0], images[len(images) // 2], images[-1]]


def _format_summary(metrics: Dict[str, np.ndarray], grok: Dict) -> List[str]:
    lines = []
    if metrics["iter"].size > 0:
        lines.append(f"- Iteracion final evaluada: {int(_latest(metrics['iter']))}")
        lines.append(f"- Train loss final: {_latest(metrics['train_loss']):.4f}")
        lines.append(f"- Val loss final: {_latest(metrics['val_loss']):.4f}")
        lines.append(f"- Grad norm global final: {_latest(metrics['global_grad_norm']):.4f}")
        lines.append(f"- Probe gap final: {_latest(metrics['probe_gap']):.4f}")
        lines.append(f"- Probe kNN purity final: {_latest(metrics['probe_knn_purity']):.4f}")
    else:
        lines.append("- No se encontraron metricas de entrenamiento.")

    if grok:
        lines.append(f"- Grokking-like heuristic: {grok.get('grokking_like', False)}")
        if "val_improvement_after_train_low" in grok:
            lines.append(
                "- Mejora val despues de train-low: "
                f"{100.0 * float(grok['val_improvement_after_train_low']):.2f}%"
            )
        if "delay_iters" in grok:
            lines.append(f"- Delay (iters): {float(grok['delay_iters']):.0f}")
    return lines


def _page_title(pdf: PdfPages, run_name: str, summary_lines: List[str]) -> None:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.9, f"PequeLLM - Training Report ({run_name})", fontsize=28, fontweight="bold")
    ax.text(0.03, 0.82, "Resumen Ejecutivo", fontsize=18, fontweight="bold")
    y = 0.75
    for line in summary_lines:
        ax.text(0.05, y, line, fontsize=14)
        y -= 0.06
    ax.text(0.03, 0.1, "Generado automaticamente por emb_gpt2.py", fontsize=11, alpha=0.8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_losses(pdf: PdfPages, metrics: Dict[str, np.ndarray]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    x = metrics["iter"]

    axs[0].plot(x, metrics["train_loss"], label="train_loss", color="#1f77b4", linewidth=2)
    axs[0].plot(x, metrics["val_loss"], label="val_loss", color="#d62728", linewidth=2)
    axs[0].set_title("Train vs Validation Loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(alpha=0.25)

    axs[1].plot(x, metrics["lr"], label="learning_rate", color="#2ca02c", linewidth=2)
    axs[1].set_title("Learning Rate Schedule")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("LR")
    axs[1].grid(alpha=0.25)

    fig.suptitle("Optimization Dynamics", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _page_gradients(pdf: PdfPages, metrics: Dict[str, np.ndarray], grad_matrix: np.ndarray, layer_names: List[str], iter_ticks: List[int]) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), height_ratios=[1.0, 1.4])
    x = metrics["iter"]

    axs[0].plot(x, metrics["global_grad_norm"], color="#9467bd", linewidth=2, label="global_grad_norm")
    axs[0].plot(x, metrics["probe_gap"], color="#ff7f0e", linewidth=2, label="probe_gap")
    axs[0].set_title("Global Gradient Norm and Probe Gap")
    axs[0].set_xlabel("Iteration")
    axs[0].grid(alpha=0.25)
    axs[0].legend()

    heat = axs[1].imshow(grad_matrix.T, aspect="auto", interpolation="nearest", cmap="viridis")
    axs[1].set_title("Per-layer Gradient Norm Heatmap (top layers)")
    axs[1].set_xlabel("Logged step index")
    axs[1].set_ylabel("Layer")
    axs[1].set_yticks(np.arange(len(layer_names)))
    axs[1].set_yticklabels(layer_names, fontsize=8)
    step_positions = np.linspace(0, max(0, len(iter_ticks) - 1), num=min(8, len(iter_ticks)), dtype=int)
    axs[1].set_xticks(step_positions)
    axs[1].set_xticklabels([str(iter_ticks[i]) for i in step_positions], rotation=20, fontsize=8)
    cbar = fig.colorbar(heat, ax=axs[1], fraction=0.02, pad=0.02)
    cbar.set_label("grad_norm")

    fig.suptitle("Gradient Analysis", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _page_umap(pdf: PdfPages, run_dir: Path, umap_images: List[Path]) -> None:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.94, "Embedding Geometry Over Iterations (UMAP/PCA)", fontsize=20, fontweight="bold")
    if not umap_images:
        ax.text(0.03, 0.84, "No UMAP snapshots found in this run.", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)
        return

    positions = [(0.03, 0.08, 0.30, 0.75), (0.35, 0.08, 0.30, 0.75), (0.67, 0.08, 0.30, 0.75)]
    for image_path, (x, y, w, h) in zip(umap_images, positions):
        img = plt.imread(str(image_path))
        sub = fig.add_axes([x, y, w, h])
        sub.imshow(img)
        sub.axis("off")
        sub.set_title(image_path.stem.replace("umap_tokens_iter_", "iter="), fontsize=11)
    ax.text(0.03, 0.02, f"Source folder: {run_dir / 'umap_snapshots'}", fontsize=10, alpha=0.8)
    pdf.savefig(fig)
    plt.close(fig)


def _page_config_and_text(pdf: PdfPages, validation: Dict, grok: Dict, sample_text: str, param_md: str) -> None:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.94, "Config Validation and Grokking Signal", fontsize=20, fontweight="bold")

    y = 0.86
    ax.text(0.03, y, "Embedding validation:", fontsize=14, fontweight="bold")
    y -= 0.05
    for key, value in validation.items():
        ax.text(0.05, y, f"- {key}: {value}", fontsize=12)
        y -= 0.04
        if y < 0.48:
            break

    y = 0.45
    ax.text(0.03, y, "Grokking heuristic:", fontsize=14, fontweight="bold")
    y -= 0.05
    if grok:
        for key, value in grok.items():
            ax.text(0.05, y, f"- {key}: {value}", fontsize=12)
            y -= 0.04
            if y < 0.28:
                break
    else:
        ax.text(0.05, y, "- No grokking heuristic file found.", fontsize=12)

    snippet = textwrap.shorten(sample_text.replace("\n", " "), width=420, placeholder=" ...")
    ax.text(0.03, 0.22, "Sample generation snippet:", fontsize=14, fontweight="bold")
    ax.text(0.05, 0.17, snippet if snippet else "(empty)", fontsize=11, wrap=True)

    md_snippet = textwrap.shorten(param_md.replace("\n", " "), width=420, placeholder=" ...")
    ax.text(0.03, 0.1, "Parameter rationale snippet:", fontsize=14, fontweight="bold")
    ax.text(0.05, 0.05, md_snippet if md_snippet else "(empty)", fontsize=10, wrap=True)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _write_markdown_summary(run_dir: Path, metrics: Dict[str, np.ndarray], validation: Dict, grok: Dict, pdf_path: Path) -> Path:
    md_path = run_dir / "presentation_summary.md"
    lines = ["# PequeLLM Training Presentation Summary", ""]
    lines.append(f"- PDF presentation: `{pdf_path}`")
    if metrics["iter"].size > 0:
        lines.append(f"- Last iter: `{int(_latest(metrics['iter']))}`")
        lines.append(f"- Train loss: `{_latest(metrics['train_loss']):.4f}`")
        lines.append(f"- Val loss: `{_latest(metrics['val_loss']):.4f}`")
        lines.append(f"- Global grad norm: `{_latest(metrics['global_grad_norm']):.4f}`")
        lines.append(f"- Probe gap: `{_latest(metrics['probe_gap']):.4f}`")
    lines.append("")
    lines.append("## Embedding validation")
    for key, value in validation.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Grokking heuristic")
    if grok:
        for key, value in grok.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No data.")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def generate_presentation(run_dir: str | Path, run_name: str = "") -> Dict[str, str]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = run_name or run_dir.name

    metrics = _load_train_metrics(run_dir)
    grad_matrix, layer_names, iter_ticks = _load_grad_heatmap(run_dir)
    validation = _load_json(run_dir / "config_validation.json")
    grok = _load_json(run_dir / "grokking_heuristic.json")
    umap_images = _collect_umap_images(run_dir)
    sample_text = (run_dir / "sample_generation.txt").read_text(encoding="utf-8") if (run_dir / "sample_generation.txt").exists() else ""
    param_md = (run_dir / "parameter_guide.md").read_text(encoding="utf-8") if (run_dir / "parameter_guide.md").exists() else ""

    summary_lines = _format_summary(metrics, grok)
    pdf_path = run_dir / f"{resolved_name}_presentation.pdf"
    with PdfPages(pdf_path) as pdf:
        _page_title(pdf, resolved_name, summary_lines)
        _page_losses(pdf, metrics)
        _page_gradients(pdf, metrics, grad_matrix, layer_names, iter_ticks)
        _page_umap(pdf, run_dir, umap_images)
        _page_config_and_text(pdf, validation, grok, sample_text, param_md)

    md_path = _write_markdown_summary(run_dir, metrics, validation, grok, pdf_path)
    return {"pdf": str(pdf_path), "markdown": str(md_path)}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF presentation from emb_gpt2 run artifacts.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to a run directory with train_metrics.csv, grad_norms_by_layer.csv, etc.")
    parser.add_argument("--run-name", type=str, default="", help="Optional display name for the report.")
    args = parser.parse_args()

    outputs = generate_presentation(run_dir=args.run_dir, run_name=args.run_name)
    print(f"presentation_pdf={outputs['pdf']}")
    print(f"summary_md={outputs['markdown']}")


if __name__ == "__main__":
    _main()
