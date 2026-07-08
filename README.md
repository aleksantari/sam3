# SAM3 + ZMQ Segmentation Server (team fork)

A fork of [facebookresearch/sam3](https://github.com/facebookresearch/sam3) that
adds a **network inference server** so a separate process — e.g. a robot-control
pipeline in a different conda env or on a different machine — can request 2D
object segmentation masks over the network **without importing SAM3 or loading
any model weights**.

SAM3 (torch + the multi-GB checkpoint) runs **only** in this server process. The
client is a ~15-line pure-`zmq`/`msgpack` shim.

- **What's added on top of upstream:** [`sam3/serving/`](sam3/serving/) — the ZMQ
  server, a reference client, and a full wire-contract handoff doc.
- **Upstream SAM3 docs** (model usage, video predictor, training, etc.) are
  preserved in [`UPSTREAM_README.md`](UPSTREAM_README.md).

---

## Prerequisites

- NVIDIA GPU with **CUDA 12.6+** (this fork is developed/tested on CUDA 12.8).
- **Conda** (Miniconda/Anaconda) and **git**.
- **Hugging Face access to the gated `facebook/sam3` checkpoint** (see step 5).

---

## Setup

### 1. Clone this fork

```bash
git clone https://github.com/aleksantari/sam3.git ~/repos/sam3
cd ~/repos/sam3
```

### 2. Create the conda environment

```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
```

### 3. Install PyTorch with CUDA support

```bash
# CUDA 12.8 build (known-good: torch 2.10.0+cu128, torchvision 0.25.0+cu128)
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

> On a different CUDA version, pick the matching wheel index from
> [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Install this package with the serving extra

```bash
# The [serve] extra adds pyzmq + msgpack + msgpack-numpy on top of SAM3.
pip install -e ".[serve]"
```

### 5. Authenticate with Hugging Face (gated checkpoint)

The `facebook/sam3` checkpoint is gated. Request access on the
[model page](https://huggingface.co/facebook/sam3), then log in so the server can
download `sam3.pt` (~3.3 GB) on first launch:

```bash
hf auth login   # paste an access token from https://huggingface.co/settings/tokens
```

The checkpoint is downloaded once and cached under `~/.cache/huggingface/`.

---

## Run the segmentation server

**Launch script** (this is the exact command we use):

```bash
bash -ic 'use_conda sam3 && cd ~/repos/sam3 && python -m sam3.serving --host 0.0.0.0 --port 5557 --device cuda'
```

> `use_conda sam3` is a local shell wrapper. If you don't have it, activate the
> env directly instead:
>
> ```bash
> conda activate sam3 && cd ~/repos/sam3 && python -m sam3.serving --host 0.0.0.0 --port 5557 --device cuda
> ```

On first launch it downloads the checkpoint; after that, startup is ~5 s. When
you see the banner it is up and **waiting for client connections** (a blocking
server is not "stuck" — stop it with `Ctrl-C`):

```
[INFO] sam3.serving.zmq_server: Loading SAM3 image model ... 
[INFO] sam3.serving.zmq_server: SAM3 ZMQ segmentation server listening on tcp://0.0.0.0:5557

=== SAM3 server READY — listening on tcp://0.0.0.0:5557 ===
    Waiting for client requests. Press Ctrl-C to stop.
```

Flags: `--host` (default `0.0.0.0`), `--port` (default `5557`), `--device`
(default `cuda`), `--checkpoint <path>` (skip the HF download / run offline),
`--confidence-threshold` (default `0.5`), `--no-warmup`.

---

## Talk to the server (client side)

The client needs **only** `pyzmq`, `msgpack`, `msgpack-numpy`, `numpy` — it does
**not** import this repo. Send a single RGB frame + a prompt (`box`, `points`, or
`text`) for one object; get back ranked 2D masks at the input resolution.

The full request/reply schema, mask dtype/shape, caveats, and a copy-paste
~15-line client shim live in **[`sam3/serving/README.md`](sam3/serving/README.md)**.

Quick health check against a running server:

```python
import msgpack, msgpack_numpy, zmq
msgpack_numpy.patch()
s = zmq.Context.instance().socket(zmq.REQ); s.connect("tcp://127.0.0.1:5557")
s.send(msgpack.packb({"action": "health"}, use_bin_type=True))
print(msgpack.unpackb(s.recv(), raw=False))   # -> {'status': 'ok'}
```

### Self-test

```bash
# Offline dispatch + loopback round-trip (no GPU / no model):
python -m pytest test/test_zmq_serving.py -v

# Full GPU round-trip (loads the real model):
SAM3_RUN_SERVE_INTEGRATION=1 python -m pytest test/test_zmq_serving.py -v
```

---

## Keeping up to date with upstream SAM3

This fork tracks `facebookresearch/sam3` as the `upstream` remote:

```bash
git fetch upstream
git merge upstream/main      # or: git rebase upstream/main
```

Push shared changes to this fork's `origin` (`aleksantari/sam3`).
