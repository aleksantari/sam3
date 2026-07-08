# SAM3 ZMQ Segmentation Server — Wire-Contract Handoff (v1)

A network boundary so a **separate** process (e.g. a robot-control pipeline in a
different conda env / on a different box) can request 2D object segmentation
masks from SAM3 without importing this repo. SAM3 (torch + the multi-GB
checkpoint) runs **only** in the server process.

SAM3 here is a **pure 2D image segmenter**: you send one RGB frame + a prompt
identifying ONE object; you get back a 2D mask in image pixel coordinates. No
depth / point clouds / 3D / camera intrinsics / robot frames — the client owns
all of that and applies the returned mask to its own pixel-aligned depth map.

This service is protocol-identical to the sibling **GraspGenX** ZMQ service
(same transport, msgpack conventions, `action` dispatch, error shape). GraspGenX
binds `5556`; this one binds **`5557`**.

---

## Launch (server side, in the `sam3` conda env)

```bash
# Install the serving extra once:
pip install -e ".[serve]"        # adds pyzmq, msgpack, msgpack-numpy

# Auto-download the checkpoint from Hugging Face (facebook/sam3):
python -m sam3.serving --host 0.0.0.0 --port 5557 --device cuda

# ...or point at a local checkpoint:
python -m sam3.serving --checkpoint /path/to/sam3.pt --device cuda
```

Flags: `--host` (default `0.0.0.0`), `--port` (default `5557`), `--checkpoint`
(default: auto-download `facebook/sam3`), `--device` (default `cuda`),
`--confidence-threshold` (default `0.5`, text path only), `--no-warmup`.

The model is loaded **once** at startup and warmed up on a synthetic frame, so
the first real request is hot. Startup prints a `Loading SAM3 image model ...`
line and then a `=== SAM3 server READY — listening on ... ===` banner; after the
banner the process **blocks waiting for client connections** (that is not a
hang — it's a long-running server, stop with Ctrl-C).

Even when the checkpoint is cached, startup makes a quick Hugging Face request to
resolve it (~3-5s with a warm cache). On an offline box, pass an explicit
`--checkpoint /path/to/sam3.pt` to skip Hugging Face entirely (or set
`HF_HUB_OFFLINE=1`).

---

## Transport

- ZMQ **REQ/REP**. Server binds `tcp://0.0.0.0:5557` (a `REP` socket); client
  uses a `REQ` socket. One in-flight request at a time.
- Body is **msgpack** with `msgpack_numpy.patch()` so numpy arrays travel
  natively. Pack with `msgpack.packb(obj, use_bin_type=True)`, unpack with
  `msgpack.unpackb(raw, raw=False)`.
- **Client dependencies: `pyzmq`, `msgpack`, `msgpack-numpy`, `numpy` — nothing
  else.** You do **not** (and must not) import the `sam3` package to talk to the
  server. (Importing `sam3` triggers torch + a checkpoint download.)

---

## Request / Reply schema

### `health`
```
->  {"action": "health"}
<-  {"status": "ok"}
```

### `metadata`
```
->  {"action": "metadata"}
<-  {"model": "sam3", "device": "cuda", "prompt_types": ["box", "points", "text"]}
```

### `segment`
```
->  {"action": "segment",
     "image":  (H, W, 3) uint8 RGB,        # full frame; mask is returned at THIS (H,W)
     "prompt": { ...exactly one of: },
         "box":    [x0, y0, x1, y1],        # pixel coords, XYXY
         "points": [[x, y], ...],           # pixel coords; optional partner field:
         "point_labels": [1, 0, ...],       #   1 = foreground, 0 = background (default: all 1)
         "text":   "red block",             # open-vocabulary concept
     "top_k": 1,                            # return the best K masks (ambiguity handling)
     "return_scores": true}

<-  {"masks":  (M, H, W) uint8, values 0 or 255, ranked best-first  (M <= top_k),
     "scores": (M,) float32, descending,
     "labels": [str] * M,                   # the text prompt for text; "" for box/points
     "timing": {"infer_ms": float}}

    or, on any true exception:
<-  {"error": "<Type>: <msg>"}              # e.g. "ValueError: empty prompt"
```

**Prompt routing (server internals, for your awareness):** `box` / `points` go
through SAM3's SAM1-style interactive predictor (pixel coords, single object,
`top_k>1` returns SAM's multi-mask ambiguity candidates). `text` goes through
SAM3's open-vocabulary grounding and returns *all* matching instances ranked by
score — `top_k` then trims to the best K.

---

## Guarantees & caveats

- **Mask resolution** = input `(H, W)`. The mask is never resized away from the
  frame you sent, so it aligns pixel-for-pixel with depth you own.
- **Mask dtype**: `uint8`, values `0` or `255`. Threshold with `mask > 0` or
  feed straight to OpenCV.
- **Object not found** (e.g. a text prompt matched nothing) is **not** an error:
  you get `masks` shaped `(0, H, W)`, `scores` shaped `(0,)`, `labels` `[]`.
  Only true exceptions take the `error` path. Always check `masks.shape[0]`.
- **Coordinates are input-image pixels** (XYXY for `box`, XY for `points`),
  origin top-left.
- **Bandwidth**: a raw `1280x720x3` frame is ~2.6 MB/request. An optional
  `image_jpeg` (JPEG bytes the server would decode) is **reserved but not
  implemented in v1** — send raw numpy. Add it later if LAN latency bites.
- One request in flight at a time (REQ/REP). Set a client `RCVTIMEO`/`SNDTIMEO`.

---

## Minimal client shim (~15 lines, copy this — imports nothing from `sam3`)

```python
import msgpack, msgpack_numpy, numpy as np, zmq
msgpack_numpy.patch()

class Sam3Client:
    def __init__(self, host="127.0.0.1", port=5557, timeout_ms=60000):
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{host}:{port}")

    def _req(self, payload):
        self.sock.send(msgpack.packb(payload, use_bin_type=True))
        rep = msgpack.unpackb(self.sock.recv(), raw=False)
        if isinstance(rep, dict) and "error" in rep:
            raise RuntimeError(rep["error"])
        return rep

    def segment(self, rgb, prompt, top_k=1):   # prompt e.g. {"box": [x0,y0,x1,y1]}
        rep = self._req({"action": "segment", "image": np.ascontiguousarray(rgb, np.uint8),
                         "prompt": prompt, "top_k": top_k, "return_scores": True})
        return rep["masks"], rep["scores"], rep["labels"]   # (M,H,W) u8, (M,) f32, [str]

# masks, scores, _ = Sam3Client().segment(frame, {"box": [320, 180, 960, 540]})
# best = masks[0]            # (H, W) uint8 0/255, pixel-aligned with your depth
```

A fuller reference client lives in [`zmq_client.py`](zmq_client.py)
(`Sam3SegmentationClient`) — but it imports `sam3.serving`, so copy the shim
above for the robot side rather than importing it.

---

## Self-test

```bash
# Stub dispatch + loopback round-trip (no model / no GPU):
python -m pytest test/test_zmq_serving.py -v

# Full GPU round-trip (downloads the checkpoint on first run):
SAM3_RUN_SERVE_INTEGRATION=1 python -m pytest test/test_zmq_serving.py -v
```
