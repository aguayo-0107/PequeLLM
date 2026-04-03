from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(env_name: str) -> bool:
    return os.getenv(env_name, "").strip().lower() in TRUTHY


def load_dataset_secure(*args: Any, trust_remote_code: bool | None = None, **kwargs: Any):
    allow_remote_code = _is_truthy("PEQUELLM_ALLOW_REMOTE_CODE")
    if trust_remote_code is None:
        trust_remote_code = allow_remote_code
    if trust_remote_code and not allow_remote_code:
        raise RuntimeError(
            "Remote dataset code is blocked by default. "
            "Set PEQUELLM_ALLOW_REMOTE_CODE=1 to opt in explicitly."
        )

    from datasets import load_dataset

    kwargs["trust_remote_code"] = bool(trust_remote_code and allow_remote_code)
    return load_dataset(*args, **kwargs)


def maybe_login_from_env() -> bool:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        return False
    if not _is_truthy("PEQUELLM_ENABLE_HF_LOGIN"):
        return False

    from huggingface_hub import login

    login(token=token)
    return True


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(target.parent),
            prefix=f"{target.name}.",
            suffix=".tmp",
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()

        os.replace(tmp_path, target)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def validate_uint16_ids(ids, *, source_name: str = "token stream") -> None:
    invalid = [int(token_id) for token_id in ids if int(token_id) < 0 or int(token_id) > 65535]
    if invalid:
        raise ValueError(
            f"{source_name} contains token IDs outside the uint16 range: "
            f"min={min(invalid)}, max={max(invalid)}"
        )
