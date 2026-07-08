# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Smoke tests for the SAM3 ZMQ serving layer.

Covers the dispatch routing and the msgpack/ZMQ wire round-trip without loading
any model weights. GPU-based inference is exercised by the integration test at
the bottom (skipped unless ``SAM3_RUN_SERVE_INTEGRATION=1``).

The whole file is skipped when the ``serve`` extras aren't installed
(``pip install -e ".[serve]"`` brings in pyzmq + msgpack + msgpack-numpy).
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

pytest.importorskip("zmq", reason="install sam3[serve] to exercise serving tests")
pytest.importorskip("msgpack", reason="install sam3[serve] to exercise serving tests")
pytest.importorskip(
    "msgpack_numpy", reason="install sam3[serve] to exercise serving tests"
)


def _stub_server():
    """Construct a Sam3SegmentationServer with __init__ bypassed.

    This skips model loading so the dispatch / validation logic can be exercised
    without checkpoints or CUDA. Only attributes actually read by ``_dispatch``,
    ``_handle_metadata`` and the validation prologue of ``_handle_segment`` are
    populated.
    """
    from sam3.serving.zmq_server import Sam3SegmentationServer

    srv = Sam3SegmentationServer.__new__(Sam3SegmentationServer)
    srv.model = None
    srv.processor = None
    srv.model_name = "sam3"
    srv.device = "cpu"
    srv.prompt_types = ["box", "points", "text"]
    return srv


def test_serving_module_imports():
    from sam3.serving import Sam3SegmentationClient, Sam3SegmentationServer

    assert Sam3SegmentationClient is not None
    assert Sam3SegmentationServer is not None


def test_dispatch_health_returns_ok():
    srv = _stub_server()
    assert srv._dispatch({"action": "health"}) == {"status": "ok"}


def test_dispatch_metadata_shape():
    srv = _stub_server()
    meta = srv._dispatch({"action": "metadata"})
    assert meta["model"] == "sam3"
    assert meta["device"] == "cpu"
    assert meta["prompt_types"] == ["box", "points", "text"]


def test_dispatch_unknown_action_raises():
    srv = _stub_server()
    with pytest.raises(ValueError, match="Unknown action"):
        srv._dispatch({"action": "bogus"})


def test_segment_empty_prompt_raises():
    srv = _stub_server()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="prompt"):
        srv._dispatch({"action": "segment", "image": img, "prompt": {}})


def test_segment_bad_image_shape_raises():
    srv = _stub_server()
    img = np.zeros((8, 8), dtype=np.uint8)  # missing channel dim
    with pytest.raises(ValueError, match="H, W, 3"):
        srv._dispatch(
            {"action": "segment", "image": img, "prompt": {"box": [0, 0, 4, 4]}}
        )


def test_segment_unsupported_prompt_key_raises():
    srv = _stub_server()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported prompt key"):
        srv._dispatch(
            {"action": "segment", "image": img, "prompt": {"mask": [1, 2, 3]}}
        )


def _run_rep_server(srv, sock, stop):
    """Serve dispatch results over a REP socket until ``stop`` is set."""
    import msgpack
    import zmq

    while not stop.is_set():
        try:
            raw = sock.recv(flags=zmq.NOBLOCK)
        except zmq.error.Again:
            time.sleep(0.01)
            continue
        try:
            req = msgpack.unpackb(raw, raw=False)
            resp = srv._dispatch(req)
        except Exception as exc:  # noqa: BLE001
            resp = {"error": f"{type(exc).__name__}: {exc}"}
        sock.send(msgpack.packb(resp, use_bin_type=True))


def test_round_trip_health_over_zmq():
    """End-to-end: bind REP, dispatch in a thread, hit it with the client."""
    import msgpack_numpy
    import zmq

    msgpack_numpy.patch()

    from sam3.serving.zmq_client import Sam3SegmentationClient

    srv = _stub_server()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    worker = threading.Thread(
        target=_run_rep_server, args=(srv, sock, stop), daemon=True
    )
    worker.start()
    try:
        with Sam3SegmentationClient(
            host="127.0.0.1", port=port, timeout_ms=5_000
        ) as client:
            assert client.health() == {"status": "ok"}
            meta = client.server_metadata
            assert meta["prompt_types"] == ["box", "points", "text"]
    finally:
        stop.set()
        worker.join(timeout=2.0)
        sock.close(linger=0)


@pytest.mark.skipif(
    os.environ.get("SAM3_RUN_SERVE_INTEGRATION") != "1",
    reason="Requires a live GPU + SAM3 checkpoint; set SAM3_RUN_SERVE_INTEGRATION=1.",
)
def test_end_to_end_segment_round_trip():
    """Full segmentation round-trip with real model weights. Off by default."""
    import msgpack_numpy
    import zmq

    msgpack_numpy.patch()

    from sam3.serving.zmq_client import Sam3SegmentationClient
    from sam3.serving.zmq_server import Sam3SegmentationServer

    device = os.environ.get("SAM3_TEST_DEVICE", "cuda")
    checkpoint = os.environ.get("SAM3_TEST_CHECKPOINT")  # None -> HF download
    srv = Sam3SegmentationServer(device=device, checkpoint=checkpoint)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    worker = threading.Thread(
        target=_run_rep_server, args=(srv, sock, stop), daemon=True
    )
    worker.start()
    try:
        h, w = 720, 1280
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        with Sam3SegmentationClient(
            host="127.0.0.1", port=port, timeout_ms=120_000
        ) as client:
            assert client.health() == {"status": "ok"}
            meta = client.server_metadata
            assert meta["model"] == "sam3"
            # box prompt over the image center
            masks, scores, labels = client.segment(
                img, box=[w // 4, h // 4, 3 * w // 4, 3 * h // 4], top_k=1
            )
            assert masks.dtype == np.uint8
            assert masks.ndim == 3 and masks.shape[1:] == (h, w)
            assert scores.shape[0] == masks.shape[0] == len(labels)
    finally:
        stop.set()
        worker.join(timeout=2.0)
        sock.close(linger=0)
