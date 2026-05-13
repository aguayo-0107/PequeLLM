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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    # Devuelve el último valor no-NaN, o NaN si todos son NaN
    valid = values[~np.isnan(values)]
    return float(valid[-1]) if valid.size > 0 else float("nan")


def _safe_col(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    """Lee una columna del CSV de forma segura — retorna array de NaNs si no existe."""
    return np.array([_to_float(r.get(key, "nan")) for r in rows], dtype=np.float64)


def _warmup_mask(metrics: Dict[str, np.ndarray], loss_threshold: float = 10.0) -> np.ndarray:
    """Genera un mask booleano que excluye los iters de arranque donde el loss es
    anormalmente alto (e.g. iter=0 con loss=455). Esto evita que la escala Y se
    comprima y haga invisible el resto de la curva.

    Se aplica a todas las métricas que se ven afectadas por el spike inicial:
    - train_loss, val_loss, perplexity, bits_per_char (derivados del loss)
    - global_grad_norm, param_norm (spikes al inicio)

    No se aplica a: probe_gap, kNN purity, lr, emb_norm (arrancan normal).

    threshold=10.0 es conservador — cualquier loss > 10 es arranque desde cero.
    Un modelo entrenado en español debería tener loss < 7 rápidamente.
    """
    val = metrics.get("val_loss", np.array([]))
    if val.size == 0:
        return np.ones(0, dtype=bool)
    return val < loss_threshold


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_train_metrics(run_dir: Path) -> Dict[str, np.ndarray]:
    rows = _read_csv(run_dir / "train_metrics.csv")

    # Columnas originales + nuevas métricas GPT-2
    all_keys = [
        "iter", "train_loss", "val_loss", "lr", "global_grad_norm",
        "probe_gap", "probe_knn_purity", "probe_intra_cos", "probe_inter_cos",
        # Nuevas
        "perplexity", "bits_per_char", "tokens_per_sec",
        "param_norm", "grad_norm_ratio", "loss_spike_count",
        "emb_norm_mean", "emb_norm_std", "emb_norm_min", "emb_norm_max",
    ]

    if not rows:
        return {key: np.array([], dtype=np.float64) for key in all_keys}

    return {key: _safe_col(rows, key) for key in all_keys}


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

    # Fill gaps con el valor anterior para que el heatmap sea legible
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


# ---------------------------------------------------------------------------
# Summary builder — ahora incluye métricas GPT-2
# ---------------------------------------------------------------------------

def _format_summary(metrics: Dict[str, np.ndarray], grok: Dict) -> List[str]:
    lines = []
    if metrics["iter"].size > 0:
        lines.append(f"- Iteracion final evaluada: {int(_latest(metrics['iter']))}")
        lines.append(f"- Train loss final: {_latest(metrics['train_loss']):.4f}")
        lines.append(f"- Val loss final: {_latest(metrics['val_loss']):.4f}")

        # Perplexity y BPC si están disponibles
        ppl = _latest(metrics["perplexity"])
        bpc = _latest(metrics["bits_per_char"])
        if not math.isnan(ppl):
            lines.append(f"- Perplexity final: {ppl:.2f}")
        if not math.isnan(bpc):
            lines.append(f"- Bits per character final: {bpc:.4f}")

        lines.append(f"- Grad norm global final: {_latest(metrics['global_grad_norm']):.4f}")

        param_norm = _latest(metrics["param_norm"])
        if not math.isnan(param_norm):
            lines.append(f"- Param norm final: {param_norm:.2f}")

        lines.append(f"- Probe gap final: {_latest(metrics['probe_gap']):.4f}")
        lines.append(f"- Probe kNN purity final: {_latest(metrics['probe_knn_purity']):.4f}")

        spikes = _latest(metrics["loss_spike_count"])
        if not math.isnan(spikes):
            lines.append(f"- Loss spikes totales: {int(spikes)}")
    else:
        lines.append("- No se encontraron metricas de entrenamiento.")

    if grok:
        lines.append(f"- Grokking-like heuristic: {grok.get('grokking_like', False)}")
        if "val_improvement_after_train_low" in grok:
            lines.append(
                f"- Mejora val despues de train-low: "
                f"{100.0 * float(grok['val_improvement_after_train_low']):.2f}%"
            )
        if "delay_iters" in grok:
            lines.append(f"- Delay (iters): {float(grok['delay_iters']):.0f}")
    return lines


# ---------------------------------------------------------------------------
# PDF pages
# ---------------------------------------------------------------------------

def _page_title(pdf: PdfPages, run_name: str, summary_lines: List[str]) -> None:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.92, f"PequeLLM - Training Report ({run_name})", fontsize=26, fontweight="bold")
    ax.text(0.03, 0.84, "Resumen Ejecutivo", fontsize=18, fontweight="bold")
    y = 0.78
    for line in summary_lines:
        ax.text(0.05, y, line, fontsize=13)
        y -= 0.055
        if y < 0.12:
            break
    ax.text(0.03, 0.05, "Generado automaticamente por emb_gpt2.py", fontsize=11, alpha=0.7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_losses(pdf: PdfPages, metrics: Dict[str, np.ndarray]) -> None:
    """Página 2: Loss + LR + Perplexity + BPC."""
    has_ppl = metrics["perplexity"].size > 0 and not np.all(np.isnan(metrics["perplexity"]))
    has_bpc = metrics["bits_per_char"].size > 0 and not np.all(np.isnan(metrics["bits_per_char"]))

    # Aplicar mask para excluir iters de arranque con loss anormalmente alto
    mask = _warmup_mask(metrics)
    x            = metrics["iter"][mask]
    train_loss   = metrics["train_loss"][mask]
    val_loss     = metrics["val_loss"][mask]
    lr           = metrics["lr"][mask]
    perplexity   = metrics["perplexity"][mask] if has_ppl else np.array([])
    bits_per_char = metrics["bits_per_char"][mask] if has_bpc else np.array([])

    # Si después del mask no queda nada, usar los datos completos
    if x.size == 0:
        x, train_loss, val_loss, lr = (metrics["iter"], metrics["train_loss"],
                                        metrics["val_loss"], metrics["lr"])
        perplexity    = metrics["perplexity"] if has_ppl else np.array([])
        bits_per_char = metrics["bits_per_char"] if has_bpc else np.array([])

    ncols = 2 + int(has_ppl) + int(has_bpc)
    fig, axs = plt.subplots(1, ncols, figsize=(16, 9))
    if ncols == 1:
        axs = [axs]
    col = 0

    # Train vs Val loss
    axs[col].plot(x, train_loss, label="train_loss", color="#1f77b4", linewidth=2)
    axs[col].plot(x, val_loss, label="val_loss", color="#d62728", linewidth=2)
    axs[col].set_title("Train vs Validation Loss")
    axs[col].set_xlabel("Iteration")
    axs[col].set_ylabel("Loss")
    axs[col].legend()
    axs[col].grid(alpha=0.25)
    col += 1

    # Learning rate — no necesita mask, el warmup es parte del experimento
    axs[col].plot(metrics["iter"], metrics["lr"], label="learning_rate", color="#2ca02c", linewidth=2)
    axs[col].set_title("Learning Rate Schedule")
    axs[col].set_xlabel("Iteration")
    axs[col].set_ylabel("LR")
    axs[col].grid(alpha=0.25)
    col += 1

    # Perplexity
    if has_ppl and perplexity.size > 0:
        axs[col].plot(x, perplexity, label="perplexity", color="#8c564b", linewidth=2)
        axs[col].set_title("Perplexity (e^val_loss)")
        axs[col].set_xlabel("Iteration")
        axs[col].set_ylabel("Perplexity")
        axs[col].grid(alpha=0.25)
        col += 1

    # Bits per character
    if has_bpc and bits_per_char.size > 0:
        axs[col].plot(x, bits_per_char, label="bits_per_char", color="#e377c2", linewidth=2)
        axs[col].set_title("Bits per Character")
        axs[col].set_xlabel("Iteration")
        axs[col].set_ylabel("BPC")
        axs[col].grid(alpha=0.25)

    fig.suptitle("Optimization Dynamics", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _page_gradients(
    pdf: PdfPages,
    metrics: Dict[str, np.ndarray],
    grad_matrix: np.ndarray,
    layer_names: List[str],
    iter_ticks: List[int],
) -> None:
    # Mask para global_grad_norm — tiene spike enorme al inicio
    mask = _warmup_mask(metrics)
    x        = metrics["iter"][mask] if mask.size > 0 else metrics["iter"]
    grad_norm = metrics["global_grad_norm"][mask] if mask.size > 0 else metrics["global_grad_norm"]
    probe_gap = metrics["probe_gap"][mask] if mask.size > 0 else metrics["probe_gap"]
    grad_ratio = metrics["grad_norm_ratio"][mask] if mask.size > 0 else metrics["grad_norm_ratio"]

    fig, axs = plt.subplots(2, 1, figsize=(16, 9), height_ratios=[1.0, 1.4])

    axs[0].plot(x, grad_norm, color="#9467bd", linewidth=2, label="global_grad_norm")
    axs[0].plot(x, probe_gap, color="#ff7f0e", linewidth=2, label="probe_gap")

    has_ratio = grad_ratio.size > 0 and not np.all(np.isnan(grad_ratio))
    if has_ratio:
        ax2 = axs[0].twinx()
        ax2.plot(x, grad_ratio, color="#17becf", linewidth=1.5,
                 linestyle="--", alpha=0.7, label="grad_norm_ratio")
        ax2.set_ylabel("grad_norm_ratio", color="#17becf", fontsize=9)
        ax2.axhline(y=3.0, color="#17becf", linestyle=":", alpha=0.5, label="spike threshold (3x)")

    axs[0].set_title("Global Gradient Norm, Probe Gap and Norm Ratio")
    axs[0].set_xlabel("Iteration")
    axs[0].grid(alpha=0.25)
    axs[0].legend(loc="upper left")

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


def _page_model_health(pdf: PdfPages, metrics: Dict[str, np.ndarray]) -> None:
    """Página nueva: salud del modelo — param_norm, embedding stats, tokens/sec.
    Solo se incluye si hay datos de las nuevas métricas."""
    has_param  = not np.all(np.isnan(metrics["param_norm"]))
    has_emb    = not np.all(np.isnan(metrics["emb_norm_mean"]))
    has_tps    = not np.all(np.isnan(metrics["tokens_per_sec"]))

    if not (has_param or has_emb or has_tps):
        return

    # Mask para param_norm y tokens_per_sec que tienen spikes al inicio
    mask = _warmup_mask(metrics)
    x_masked = metrics["iter"][mask] if mask.size > 0 else metrics["iter"]
    x_full   = metrics["iter"]

    plots = []
    if has_param:
        data = metrics["param_norm"][mask] if mask.size > 0 else metrics["param_norm"]
        plots.append(("param_norm", x_masked, data, "#e74c3c",
                      "Param Norm (L2 total de pesos)",
                      "Si crece sin control → pesos explotando"))
    if has_emb:
        # emb_norm arranca normal, no necesita mask
        plots.append(("emb_norm_mean", x_full, metrics["emb_norm_mean"], "#3498db",
                      "Embedding Norm Media",
                      "Debe crecer gradualmente conforme aprende"))
    if has_tps:
        # tokens_per_sec: spikes de UMAP distorsionan, aplicar mask
        data = metrics["tokens_per_sec"][mask] if mask.size > 0 else metrics["tokens_per_sec"]
        plots.append(("tokens_per_sec", x_masked, data, "#27ae60",
                      "Tokens por Segundo",
                      "Eficiencia de GPU — caídas = throttling"))

    fig, axs = plt.subplots(1, len(plots), figsize=(16, 9))
    if len(plots) == 1:
        axs = [axs]

    for ax, (key, xdata, data, color, title, subtitle) in zip(axs, plots):
        ax.plot(xdata, data, color=color, linewidth=2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(key)
        ax.text(0.5, -0.15, subtitle, transform=ax.transAxes,
                ha="center", fontsize=9, alpha=0.7)
        ax.grid(alpha=0.25)

    # Banda ±1std para embedding norm
    if has_emb:
        emb_ax = axs[[k for k, (key, *_) in enumerate(plots) if key == "emb_norm_mean"][0]]
        std  = metrics["emb_norm_std"]
        mean = metrics["emb_norm_mean"]
        valid = ~(np.isnan(mean) | np.isnan(std))
        if valid.any():
            emb_ax.fill_between(
                x_full[valid], (mean - std)[valid], (mean + std)[valid],
                alpha=0.25, color="#3498db", label="±1 std"
            )
            emb_ax.legend()

    fig.suptitle("Model Health Metrics", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _page_probe_semantics(pdf: PdfPages, metrics: Dict[str, np.ndarray]) -> None:
    """Página nueva: métricas semánticas — probe_gap, knn_purity, intra/inter cosine."""
    x = metrics["iter"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 9))

    # probe_gap
    axs[0].plot(x, metrics["probe_gap"], color="#e67e22", linewidth=2)
    axs[0].set_title("Probe Gap\n(intra_cos − inter_cos)", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("gap")
    axs[0].grid(alpha=0.25)
    axs[0].text(0.5, -0.15,
                "Más alto = palabras similares más cerca en el espacio vectorial",
                transform=axs[0].transAxes, ha="center", fontsize=9, alpha=0.7)

    # kNN purity
    axs[1].plot(x, metrics["probe_knn_purity"], color="#8e44ad", linewidth=2)
    axs[1].set_title("kNN Purity\n(vecinos de misma categoría)", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("purity [0-1]")
    axs[1].set_ylim(0, 1.05)
    axs[1].grid(alpha=0.25)
    axs[1].text(0.5, -0.15,
                "1.0 = todos los vecinos son de la misma categoría semántica",
                transform=axs[1].transAxes, ha="center", fontsize=9, alpha=0.7)

    # intra vs inter cosine similarity
    has_intra = not np.all(np.isnan(metrics["probe_intra_cos"]))
    has_inter = not np.all(np.isnan(metrics["probe_inter_cos"]))
    if has_intra:
        axs[2].plot(x, metrics["probe_intra_cos"], color="#27ae60", linewidth=2, label="intra (misma cat.)")
    if has_inter:
        axs[2].plot(x, metrics["probe_inter_cos"], color="#e74c3c", linewidth=2, label="inter (distinta cat.)")
    axs[2].set_title("Cosine Similarity\nIntra vs Inter categoría", fontsize=12, fontweight="bold")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("cosine similarity")
    axs[2].legend()
    axs[2].grid(alpha=0.25)
    axs[2].text(0.5, -0.15,
                "Brecha entre intra e inter = separación semántica aprendida",
                transform=axs[2].transAxes, ha="center", fontsize=9, alpha=0.7)

    fig.suptitle("Semantic Probe Metrics (PROBE_WORDS)", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _page_umap(pdf: PdfPages, run_dir: Path, umap_images: List[Path]) -> None:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.94, "Embedding Geometry Over Iterations (UMAP/PCA)",
            fontsize=20, fontweight="bold")
    if not umap_images:
        ax.text(0.03, 0.84, "No UMAP snapshots found in this run.", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)
        return

    positions = [(0.03, 0.08, 0.29, 0.78), (0.35, 0.08, 0.29, 0.78), (0.67, 0.08, 0.29, 0.78)]
    for image_path, (x, y, w, h) in zip(umap_images, positions):
        img = plt.imread(str(image_path))
        sub = fig.add_axes([x, y, w, h])
        sub.imshow(img)
        sub.axis("off")
        sub.set_title(image_path.stem.replace("umap_tokens_iter_", "iter="), fontsize=11)
    ax.text(0.03, 0.02, f"Source folder: {run_dir / 'umap_snapshots'}", fontsize=10, alpha=0.7)
    pdf.savefig(fig)
    plt.close(fig)


def _page_config_and_text(
    pdf: PdfPages, validation: Dict, grok: Dict, sample_text: str, param_md: str
) -> None:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.94, "Config Validation and Grokking Signal",
            fontsize=20, fontweight="bold")

    y = 0.86
    ax.text(0.03, y, "Embedding validation:", fontsize=14, fontweight="bold")
    y -= 0.05
    for key, value in validation.items():
        ax.text(0.05, y, f"- {key}: {value}", fontsize=12)
        y -= 0.04
        if y < 0.50:
            break

    y = 0.47
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
    ax.text(0.03, 0.10, "Parameter rationale snippet:", fontsize=14, fontweight="bold")
    ax.text(0.05, 0.05, md_snippet if md_snippet else "(empty)", fontsize=10, wrap=True)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary — ahora incluye nuevas métricas
# ---------------------------------------------------------------------------

def _write_markdown_summary(
    run_dir: Path, metrics: Dict[str, np.ndarray], validation: Dict, grok: Dict, pdf_path: Path
) -> Path:
    md_path = run_dir / "presentation_summary.md"
    lines = ["# PequeLLM Training Presentation Summary", ""]
    lines.append(f"- PDF presentation: `{pdf_path}`")

    if metrics["iter"].size > 0:
        lines.append(f"- Last iter: `{int(_latest(metrics['iter']))}`")
        lines.append(f"- Train loss: `{_latest(metrics['train_loss']):.4f}`")
        lines.append(f"- Val loss: `{_latest(metrics['val_loss']):.4f}`")

        ppl = _latest(metrics["perplexity"])
        if not math.isnan(ppl):
            lines.append(f"- Perplexity: `{ppl:.2f}`")

        bpc = _latest(metrics["bits_per_char"])
        if not math.isnan(bpc):
            lines.append(f"- Bits per character: `{bpc:.4f}`")

        lines.append(f"- Global grad norm: `{_latest(metrics['global_grad_norm']):.4f}`")

        pnorm = _latest(metrics["param_norm"])
        if not math.isnan(pnorm):
            lines.append(f"- Param norm: `{pnorm:.2f}`")

        lines.append(f"- Probe gap: `{_latest(metrics['probe_gap']):.4f}`")
        lines.append(f"- kNN purity: `{_latest(metrics['probe_knn_purity']):.4f}`")

        spikes = _latest(metrics["loss_spike_count"])
        if not math.isnan(spikes):
            lines.append(f"- Loss spikes: `{int(spikes)}`")

        emb_mean = _latest(metrics["emb_norm_mean"])
        if not math.isnan(emb_mean):
            lines.append(f"- Embedding norm mean: `{emb_mean:.4f}`")

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_presentation(run_dir: str | Path, run_name: str = "") -> Dict[str, str]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = run_name or run_dir.name

    metrics      = _load_train_metrics(run_dir)
    grad_matrix, layer_names, iter_ticks = _load_grad_heatmap(run_dir)
    validation   = _load_json(run_dir / "config_validation.json")
    grok         = _load_json(run_dir / "grokking_heuristic.json")
    umap_images  = _collect_umap_images(run_dir)
    sample_text  = (run_dir / "sample_generation.txt").read_text(encoding="utf-8") \
                   if (run_dir / "sample_generation.txt").exists() else ""
    param_md     = (run_dir / "parameter_guide.md").read_text(encoding="utf-8") \
                   if (run_dir / "parameter_guide.md").exists() else ""

    summary_lines = _format_summary(metrics, grok)
    pdf_path = run_dir / f"{resolved_name}_presentation.pdf"

    with PdfPages(pdf_path) as pdf:
        _page_title(pdf, resolved_name, summary_lines)
        _page_losses(pdf, metrics)                                          # Pág 2: loss + LR + ppl + bpc
        _page_gradients(pdf, metrics, grad_matrix, layer_names, iter_ticks) # Pág 3: gradientes + ratio
        _page_model_health(pdf, metrics)                                    # Pág 4: param_norm + emb + tps (nueva)
        _page_probe_semantics(pdf, metrics)                                 # Pág 5: probe semántico (nueva)
        _page_umap(pdf, run_dir, umap_images)                               # Pág 6: UMAP
        _page_config_and_text(pdf, validation, grok, sample_text, param_md) # Pág 7: config + texto

    md_path = _write_markdown_summary(run_dir, metrics, validation, grok, pdf_path)
    return {"pdf": str(pdf_path), "markdown": str(md_path)}


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PDF presentation from emb_gpt2 run artifacts."
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to a run directory with train_metrics.csv, grad_norms_by_layer.csv, etc.")
    parser.add_argument("--run-name", type=str, default="",
                        help="Optional display name for the report.")
    args = parser.parse_args()

    outputs = generate_presentation(run_dir=args.run_dir, run_name=args.run_name)
    print(f"presentation_pdf={outputs['pdf']}")
    print(f"summary_md={outputs['markdown']}")


if __name__ == "__main__":
    _main()