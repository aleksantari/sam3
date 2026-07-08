# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Launch a SAM3 ZMQ image-segmentation server.

Usage::

    # Auto-download the checkpoint from Hugging Face (facebook/sam3):
    python -m sam3.serving --host 0.0.0.0 --port 5557 --device cuda

    # Use a local checkpoint:
    python -m sam3.serving --checkpoint /path/to/sam3.pt --device cuda
"""

import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start a SAM3 ZMQ image-segmentation server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Address to bind the ZMQ socket (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5557,
        help="Port to bind the ZMQ socket (default: 5557)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to sam3.pt. If omitted, auto-download from facebook/sam3.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Score threshold for the text-grounding path (default: 0.5)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the synthetic warmup inference at startup.",
    )
    return parser.parse_args()


def main():
    # parse_args() first so --help exits before the heavy torch/SAM3 import.
    args = parse_args()

    # Deferred import so heavy torch/SAM3 loading happens only at launch.
    from sam3.serving.zmq_server import Sam3SegmentationServer

    # Configure logging AFTER the heavy import: torch/timm/hf reconfigure the
    # root logger on import and would otherwise swallow our INFO lines.
    # force=True removes any handlers they installed and reinstalls ours.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    # Quiet the per-startup HF/HTTP request chatter; keep our own INFO logs.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    server = Sam3SegmentationServer(
        host=args.host,
        port=args.port,
        checkpoint=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        warmup=not args.no_warmup,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
