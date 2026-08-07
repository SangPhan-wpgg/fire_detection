"""Tương thích đầu ra Unicode trên terminal Windows và môi trường redirect."""

from __future__ import annotations

import sys


def configure_utf8_output() -> None:
    """Ưu tiên UTF-8 cho stdout/stderr nếu stream hỗ trợ reconfigure."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
