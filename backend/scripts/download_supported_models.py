#!/usr/bin/env python
"""
Vita AI – Download Supported Models
=====================================
CLI script to download and cache all auto-downloadable pretrained
models into ``models_cache/``.

Usage
-----
::

    python scripts/download_supported_models.py
    python scripts/download_supported_models.py --model distilbert
    python scripts/download_supported_models.py --model biobert
    python scripts/download_supported_models.py --model yamnet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("vita_download")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download supported pretrained models for Vita AI.",
    )
    parser.add_argument(
        "--model",
        choices=["distilbert", "biobert", "yamnet", "all"],
        default="all",
        help="Which model to download (default: all).",
    )
    args = parser.parse_args()

    # Force auto-download on regardless of env
    os.environ["VITA_AUTO_DOWNLOAD_MODELS"] = "true"
    os.environ["VITA_ENABLE_DISTILBERT"] = "true"
    os.environ["VITA_ENABLE_BIOBERT"] = "true"
    os.environ["VITA_ENABLE_YAMNET"] = "true"

    from backend.app.ml.registry.model_download import (
        download_all_supported,
        download_biobert,
        download_distilbert,
        download_yamnet,
    )
    from backend.app.ml.registry.model_paths import is_model_cached

    if args.model == "all":
        logger.info("Downloading all supported models …")
        results = download_all_supported()
        if results:
            logger.info("Successfully downloaded: %s", results)
        else:
            logger.info("No models downloaded (all already cached or download failed).")
    else:
        name = args.model
        if is_model_cached(name):
            logger.info("%s is already cached.", name)
        else:
            logger.info("Downloading %s …", name)
            func = {"distilbert": download_distilbert, "biobert": download_biobert, "yamnet": download_yamnet}[name]
            ok = func()
            if ok:
                logger.info("%s downloaded successfully.", name)
            else:
                logger.error("Failed to download %s.", name)
                sys.exit(1)

    # Print summary
    print("\n" + "=" * 50)
    print("  Model Cache Status")
    print("=" * 50)

    print("\n  Auto-Downloadable Models:")
    for model in ("distilbert", "biobert", "yamnet"):
        cached = is_model_cached(model)
        status = "\033[32mCACHED\033[0m" if cached else "\033[31mNOT CACHED\033[0m"
        print(f"    {model:12s}  {status}")

    print()
    print("  To run the backend:")
    print("    uvicorn api.main:app --reload")
    print()


if __name__ == "__main__":
    main()
