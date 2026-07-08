# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""ZMQ REQ/REP server that wraps SAM3 image segmentation for remote inference.

This mirrors the sibling GraspGenX serving layer so both services are
protocol-identical: ZMQ REQ/REP, msgpack with ``msgpack_numpy.patch()`` (numpy
arrays travel natively), an ``action`` dispatch, and an
``{"error": "<Type>: <msg>"}`` reply on any exception.

SAM3 is a *pure 2D image segmenter* here: in = one RGB frame + a prompt for ONE
object, out = mask(s) in input-image pixel coordinates. No depth / 3D / camera
intrinsics / robot frames — the caller owns all of that.

Wire protocol (see ``README.md`` for the full handoff spec):

* ``{"action": "health"}`` -> ``{"status": "ok"}``
* ``{"action": "metadata"}`` ->
  ``{"model": str, "device": str, "prompt_types": ["box", "points", "text"]}``
* ``{"action": "segment",
       "image": (H, W, 3) uint8 RGB,
       "prompt": {one of "box": [x0,y0,x1,y1] px
                       | "points": [[x,y], ...], "point_labels": [1,0,...]
                       | "text": "red block"},
       "top_k": 1, "return_scores": true}`` ->
  ``{"masks": (M, H, W) uint8 0/255, "scores": (M,) float32,
     "labels": [str], "timing": {"infer_ms": float}}``

Any unhandled error is returned as ``{"error": "<Type>: <msg>"}`` — the client
raises.

IMPORTANT: the heavy imports (torch, SAM3, PIL) are deferred into
:meth:`Sam3SegmentationServer._load_model` so that importing this module — or
the :class:`Sam3SegmentationServer` class — does NOT pull in torch or trigger a
checkpoint download. Only constructing the server (or launching via
``python -m sam3.serving``) loads the model.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()

logger = logging.getLogger(__name__)


class Sam3SegmentationServer:
    """ZMQ REQ/REP wrapper around the SAM3 image model.

    The model is loaded once at construction and warmed up on a synthetic frame
    so the first real request is not slow.

    Args:
        host: ZMQ bind address (default ``0.0.0.0``).
        port: ZMQ bind port (default ``5557``; GraspGenX uses 5556).
        checkpoint: Local path to ``sam3.pt``. If ``None``, the model is
            auto-downloaded from Hugging Face (``facebook/sam3``) on first load.
        device: Torch device string (``cuda`` or ``cpu``).
        confidence_threshold: Score threshold for the text-grounding path.
        bpe_path: Optional path to the BPE tokenizer vocab (defaults to the one
            packaged with ``sam3``).
        warmup: If True (default), run one synthetic inference at startup.
    """

    SUPPORTED_PROMPT_TYPES: List[str] = ["box", "points", "text"]

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5557,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        bpe_path: Optional[str] = None,
        warmup: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.checkpoint = checkpoint
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.bpe_path = bpe_path
        self.prompt_types = list(self.SUPPORTED_PROMPT_TYPES)
        self.model_name = "sam3"

        # Heavy: builds the model and (optionally) downloads the checkpoint.
        self.model = None
        self.processor = None
        self._load_model()
        if warmup:
            self._warmup()

    # ------------------------------------------------------------------ #
    # Model lifecycle (heavy imports isolated here)
    # ------------------------------------------------------------------ #
    def _load_model(self) -> None:
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # bf16 autocast is applied per-request in _inference_context() (so it is
        # active on whatever thread runs inference, not just the load thread).
        self._use_autocast = False
        if self.device == "cuda" and torch.cuda.is_available():
            self._use_autocast = True
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        logger.info(
            "Loading SAM3 image model (checkpoint=%s, device=%s) — this can "
            "take ~30-90s the first time (model build + 3.3GB checkpoint) ...",
            self.checkpoint or "<auto-download facebook/sam3>",
            self.device,
        )
        self.model = build_sam3_image_model(
            bpe_path=self.bpe_path,
            device=self.device,
            checkpoint_path=self.checkpoint,
            enable_inst_interactivity=True,  # enables box/points predict_inst path
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        logger.info("SAM3 image model ready.")

    def _inference_context(self):
        """bf16 autocast on CUDA (active on the calling thread), else a no-op."""
        import contextlib

        if getattr(self, "_use_autocast", False):
            import torch

            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _warmup(self) -> None:
        """Run one synthetic box prediction so the first real request is hot."""
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._handle_segment(
                {"image": dummy, "prompt": {"box": [8, 8, 56, 56]}, "top_k": 1}
            )
            logger.info("Warmup inference complete.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warmup inference failed (continuing): %s", exc)

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    def _dispatch(self, request: dict) -> dict:
        action = request.get("action")
        if action == "health":
            return {"status": "ok"}
        if action == "metadata":
            return self._handle_metadata()
        if action == "segment":
            return self._handle_segment(request)
        raise ValueError(f"Unknown action: {action!r}")

    def _handle_metadata(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "prompt_types": self.prompt_types,
        }

    def _handle_segment(self, request: dict) -> dict:
        # --- validate image (done before touching the model) ---
        image = request.get("image")
        if image is None:
            raise ValueError("Request is missing 'image'.")
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must be (H, W, 3) uint8 RGB; got shape {image.shape}"
            )
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        h, w = int(image.shape[0]), int(image.shape[1])

        # --- validate prompt ---
        prompt = request.get("prompt")
        if not isinstance(prompt, dict) or not prompt:
            raise ValueError("Request has a missing or empty 'prompt'.")
        unknown = set(prompt) - {"box", "points", "point_labels", "text"}
        if unknown:
            raise ValueError(f"Unsupported prompt key(s): {sorted(unknown)}")
        has_geom = ("box" in prompt) or ("points" in prompt)
        has_text = "text" in prompt and prompt["text"]
        if not has_geom and not has_text:
            raise ValueError(
                "prompt must contain one of: 'box', 'points', or 'text'."
            )

        top_k = int(request.get("top_k", 1))

        t0 = time.monotonic()
        with self._inference_context():
            if has_text:
                masks, scores, label_text = self._segment_text(
                    image, prompt["text"]
                )
            else:
                masks, scores = self._segment_geometric(image, prompt, top_k)
                label_text = ""
        infer_ms = (time.monotonic() - t0) * 1000.0

        # rank best-first, take top_k
        masks, scores = self._rank_and_topk(masks, scores, top_k, h, w)
        labels = [label_text] * masks.shape[0]

        return {
            "masks": np.ascontiguousarray(masks),
            "scores": scores.astype(np.float32),
            "labels": labels,
            "timing": {"infer_ms": float(infer_ms)},
        }

    # ------------------------------------------------------------------ #
    # Prompt-type backends
    # ------------------------------------------------------------------ #
    def _segment_geometric(self, image: np.ndarray, prompt: dict, top_k: int):
        """Box / points via the SAM1-style ``predict_inst`` path (pixel coords)."""
        from PIL import Image

        state = self.processor.set_image(Image.fromarray(image))

        kwargs: Dict[str, object] = {"multimask_output": top_k > 1}
        if "points" in prompt:
            pts = np.asarray(prompt["points"], dtype=np.float32).reshape(-1, 2)
            if "point_labels" in prompt and prompt["point_labels"] is not None:
                labels = np.asarray(prompt["point_labels"], dtype=np.int32)
            else:
                labels = np.ones(pts.shape[0], dtype=np.int32)
            kwargs["point_coords"] = pts
            kwargs["point_labels"] = labels
        if "box" in prompt:
            kwargs["box"] = np.asarray(prompt["box"], dtype=np.float32).reshape(4)

        masks, scores, _ = self.model.predict_inst(state, **kwargs)
        # masks: (C, H, W) float 0/1 @ orig res; scores: (C,)
        return np.asarray(masks), np.asarray(scores)

    def _segment_text(self, image: np.ndarray, text: str):
        """Open-vocabulary text grounding via ``Sam3Processor.set_text_prompt``."""
        from PIL import Image

        state = self.processor.set_image(Image.fromarray(image))
        self.processor.reset_all_prompts(state)
        state = self.processor.set_text_prompt(prompt=text, state=state)

        masks_t = state.get("masks")
        scores_t = state.get("scores")
        if masks_t is None or len(masks_t) == 0:
            h, w = image.shape[0], image.shape[1]
            return np.zeros((0, h, w), dtype=np.uint8), np.zeros((0,), np.float32), text
        # .float() first: under bf16 autocast scores is BFloat16, which numpy
        # cannot convert. masks is already bool.
        masks = masks_t.detach().cpu().numpy()
        # grounding masks come back as (N, 1, H, W); drop the channel dim so the
        # wire shape matches the geometric path's (M, H, W).
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        scores = scores_t.detach().float().cpu().numpy()
        return masks, scores, text

    @staticmethod
    def _rank_and_topk(masks, scores, top_k: int, h: int, w: int):
        """Sort by score desc, keep top_k, return masks as (M, H, W) uint8 0/255."""
        masks = np.asarray(masks)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if masks.shape[0] == 0:
            return np.zeros((0, h, w), dtype=np.uint8), scores
        order = np.argsort(scores)[::-1]
        if top_k > 0:
            order = order[:top_k]
        masks = masks[order]
        scores = scores[order]
        masks_u8 = (np.asarray(masks) > 0).astype(np.uint8) * 255
        return masks_u8, scores

    # ------------------------------------------------------------------ #
    # Serve loop (copied from the GraspGenX pattern)
    # ------------------------------------------------------------------ #
    def serve_forever(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        # One in-flight request at a time (REQ/REP); drop backpressure sanely.
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.setsockopt(zmq.SNDHWM, 1)
        sock.setsockopt(zmq.LINGER, 0)
        addr = f"tcp://{self.host}:{self.port}"
        sock.bind(addr)
        logger.info("SAM3 ZMQ segmentation server listening on %s", addr)
        # Plain flushed banner so it is unmistakable the server is up and
        # *waiting* for requests (a blocking server is not "stuck").
        print(
            f"\n=== SAM3 server READY — listening on {addr} ===\n"
            f"    Waiting for client requests. Press Ctrl-C to stop.\n",
            flush=True,
        )
        try:
            while True:
                raw = sock.recv()
                try:
                    request = msgpack.unpackb(raw, raw=False)
                    response = self._dispatch(request)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Request failed: %s", exc)
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                sock.send(msgpack.packb(response, use_bin_type=True))
        except KeyboardInterrupt:
            logger.info("Shutting down SAM3 ZMQ server (KeyboardInterrupt).")
        finally:
            sock.close(linger=0)
