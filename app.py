"""ResearchMind AI entry point."""
from __future__ import annotations

import logging
import os
import threading

from config import Settings, setup_logging
from rag.services import AppContext

log = logging.getLogger("researchmind")


def create_app():
    settings = Settings()
    setup_logging(settings.log_level)
    ctx = AppContext(settings)
    if settings.preload_models:
        threading.Thread(target=ctx.warmup, daemon=True).start()
    from ui.app_ui import build_ui
    return build_ui(ctx).queue(default_concurrency_limit=2)


if __name__ == "__main__":
    create_app().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), show_error=True)
