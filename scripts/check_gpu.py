"""Smoke test: ¿el contenedor ve la GPU AMD via ROCm?

Codigos de salida:
  0 -> torch importa, CUDA disponible, matmul corre en cuda:0.
  2 -> torch importa pero torch.cuda.is_available() es False.
  3 -> torch importa, CUDA disponible, pero la operacion en GPU fallo.
  4 -> torch ni siquiera importa.

Cuando algo falla, imprime un bloque de triage para que el estudiante
sepa que mirar en docs/STUDENT.md ("GPU no visible").
"""

from __future__ import annotations

import os
import sys
import traceback


HOST_DEVICES = ["/dev/kfd", "/dev/dri", "/dev/dri/renderD128", "/dev/dri/card0"]


def _print_triage(message: str) -> None:
    print("=" * 70, file=sys.stderr)
    print("[GPU CHECK] FAILED:", message, file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("Dispositivos visibles dentro del contenedor:", file=sys.stderr)
    for path in HOST_DEVICES:
        marker = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{marker}] {path}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Banderas que run.sh debe pasar a podman/docker:", file=sys.stderr)
    print("  --device /dev/kfd --device /dev/dri \\", file=sys.stderr)
    print("  --group-add keep-groups --security-opt label=disable", file=sys.stderr)
    print("", file=sys.stderr)
    print("Mas detalle: docs/STUDENT.md, seccion 'GPU no visible'.", file=sys.stderr)
    print("=" * 70, file=sys.stderr)


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"[GPU CHECK] no se pudo importar torch: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 4

    print(f"[GPU CHECK] torch.__version__   = {torch.__version__}")
    print(f"[GPU CHECK] torch.version.hip   = {getattr(torch.version, 'hip', None)}")
    print(f"[GPU CHECK] torch.version.cuda  = {getattr(torch.version, 'cuda', None)}")

    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        arch_list = []
    print(f"[GPU CHECK] arch list           = {arch_list}")

    available = torch.cuda.is_available()
    print(f"[GPU CHECK] cuda.is_available() = {available}")
    print(f"[GPU CHECK] cuda.device_count() = {torch.cuda.device_count() if available else 0}")

    if not available:
        _print_triage(
            "torch.cuda.is_available() == False. "
            "Probable causa: faltan los devices /dev/kfd o /dev/dri, "
            "o el wheel de torch no esta compilado contra ROCm."
        )
        return 2

    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        device_name = f"<error: {exc}>"
    print(f"[GPU CHECK] device 0 name       = {device_name}")

    try:
        a = torch.randn(256, 256, device="cuda:0", dtype=torch.float32)
        b = torch.randn(256, 256, device="cuda:0", dtype=torch.float32)
        c = (a @ b).sum().item()
        if not (c == c):  # NaN check sin importar math.isnan
            raise RuntimeError(f"matmul produjo NaN: {c}")
        print(f"[GPU CHECK] matmul 256x256 OK (suma={c:.4f})")
    except Exception as exc:
        _print_triage(f"matmul en cuda:0 fallo: {exc}")
        traceback.print_exc()
        return 3

    print("[GPU CHECK] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
