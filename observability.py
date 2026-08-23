"""Small, JSON-serializable router observability helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def router_report(model) -> Dict[str, Any]:
    """Return the latest per-layer router summaries from a Transformer."""
    layers = model.router_stats()
    return {
        "schema_version": 1,
        "layers": layers,
        "captured_layers": len(layers),
    }


def write_router_report(model, path: str | Path) -> Dict[str, Any]:
    report = router_report(model)
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
