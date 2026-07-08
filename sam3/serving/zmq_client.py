# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Thin ZMQ REQ client for :class:`Sam3SegmentationServer`.

This module deliberately has NO dependency on torch, the SAM3 model weights, or
anything else in the ``sam3`` package — it is a pure msgpack/ZMQ wire-protocol
shim. It doubles as the reference implementation the robot side copies: the
client only needs ``pyzmq``, ``msgpack``, ``msgpack-numpy`` and ``numpy``.

See :mod:`sam3.serving.zmq_server` (and ``README.md``) for the protocol.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import msgpack
import msgpack_numpy
import numpy as np
import zmq

msgpack_numpy.patch()


class Sam3SegmentationClient:
    """ZMQ REQ client that round-trips msgpack payloads to a SAM3 server.

    Usage::

        with Sam3SegmentationClient(host="localhost", port=5557) as client:
            print(client.server_metadata)
            masks, scores, labels = client.segment(rgb, box=[x0, y0, x1, y1])

    Args:
        host: Server hostname (default ``localhost``).
        port: Server port (default ``5557``).
        timeout_ms: Per-request send/recv timeout in milliseconds. ``None``
            disables timeouts (the request blocks until the server replies).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5557,
        timeout_ms: Optional[int] = 60_000,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None
        self._metadata_cache: Optional[dict] = None

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def __enter__(self) -> "Sam3SegmentationClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self._sock is not None:
            return
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        if self.timeout_ms is not None:
            self._sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_ms))
            self._sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_ms))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.address)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    def _request(self, payload: dict) -> dict:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        try:
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except zmq.error.Again as exc:
            # Socket is in a bad state after a timeout — reset so callers retry.
            self.close()
            raise TimeoutError(
                f"SAM3 server at {self.address} did not respond within "
                f"{self.timeout_ms} ms"
            ) from exc
        response = msgpack.unpackb(raw, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"SAM3 server error: {response['error']}")
        return response

    @property
    def server_metadata(self) -> dict:
        """Cached metadata response. Re-fetched once per client lifetime."""
        if self._metadata_cache is None:
            self._metadata_cache = self._request({"action": "metadata"})
        return self._metadata_cache

    def health(self) -> dict:
        return self._request({"action": "health"})

    def segment(
        self,
        image: np.ndarray,
        *,
        box: Optional[List[float]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        text: Optional[str] = None,
        top_k: int = 1,
        return_scores: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Request a segmentation mask for ONE object.

        Provide exactly one prompt type: ``box`` ([x0,y0,x1,y1] pixels),
        ``points`` ([[x,y], ...] pixels, optional ``point_labels`` 1=fg/0=bg),
        or ``text`` (open-vocabulary concept).

        Returns:
            (masks, scores, labels) — masks is (M, H, W) uint8 0/255 ranked
            best-first (M <= top_k; M may be 0 if the object was not found),
            scores is (M,) float32 descending, labels is a list of M strings.
        """
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must be (H, W, 3) uint8 RGB; got shape {image.shape}"
            )

        prompt: dict = {}
        if box is not None:
            prompt["box"] = list(box)
        if points is not None:
            prompt["points"] = [list(p) for p in points]
            if point_labels is not None:
                prompt["point_labels"] = list(point_labels)
        if text is not None:
            prompt["text"] = text
        if not prompt:
            raise ValueError("Provide one of: box, points, or text.")

        response = self._request(
            {
                "action": "segment",
                "image": image,
                "prompt": prompt,
                "top_k": int(top_k),
                "return_scores": bool(return_scores),
            }
        )
        masks = np.asarray(response["masks"], dtype=np.uint8)
        scores = np.asarray(response["scores"], dtype=np.float32)
        labels = list(response.get("labels", []))
        return masks, scores, labels
