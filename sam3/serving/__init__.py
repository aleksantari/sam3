# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""ZMQ-based inference serving layer for SAM3 image segmentation.

Exposes a REQ/REP wire protocol (msgpack + msgpack-numpy) so lightweight
clients — including a robot-control process in a different env — can request 2D
segmentation masks without loading any model weights themselves.

Importing :class:`Sam3SegmentationServer` is cheap: its torch/SAM3 imports are
deferred into the constructor, so ``from sam3.serving import ...`` does not load
torch or trigger a checkpoint download.
"""

from sam3.serving.zmq_client import Sam3SegmentationClient
from sam3.serving.zmq_server import Sam3SegmentationServer

__all__ = ["Sam3SegmentationClient", "Sam3SegmentationServer"]
