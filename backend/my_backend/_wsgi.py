"""WSGI entry point for the YOLO+SAM Label Studio ML backend."""

from __future__ import annotations

# Compatibility patches MUST be applied before any ML imports.
from compat import apply_patches

apply_patches()

import logging
import os

from label_studio_ml.api import init_app

from config import get_settings
from model import YoloSamBackend

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the Label Studio ML app
app = init_app(
    model_class=YoloSamBackend,
    model_dir=os.path.dirname(os.path.abspath(__file__)),
)

# Optional request logging (controlled by DEBUG_REQUESTS env var)
if settings.debug_requests:
    from flask import request as flask_request

    @app.before_request
    def log_request() -> None:
        logger.debug(
            "REQUEST %s %s content_type=%s",
            flask_request.method,
            flask_request.path,
            flask_request.content_type,
        )


if __name__ == "__main__":
    logger.info(
        "YOLO+SAM Backend starting on http://%s:%d",
        settings.ml_backend_host,
        settings.ml_backend_port,
    )
    app.run(
        host=settings.ml_backend_host,
        port=settings.ml_backend_port,
        debug=False,
    )
