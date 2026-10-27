import json

import numpy as np
import torch

from data.depth_encoding.multiscale_disparity_transform import DepthMultiScaleDisparityTransform
from train.rl.geometry_tools.engine_utils.runtime_v3 import SelfContainedToolSession


def test_depth_codec_round_trip_in_meters():
    depth = torch.logspace(-1, 2, 64, dtype=torch.float64).reshape(1, 1, 8, 8)
    codec = DepthMultiScaleDisparityTransform()
    encoded = codec(depth)
    assert encoded.shape == (1, 3, 8, 8)
    torch.testing.assert_close(codec.decode(encoded), depth.squeeze(1), rtol=1e-5, atol=1e-6)


def test_bundled_tool_reads_known_metric_depth():
    depth = np.full((64, 64), 3.0, dtype=np.float32)
    session = SelfContainedToolSession(
        depth=depth, conf=np.ones_like(depth), intrinsics=np.eye(3),
        image_wh=(64, 64), conf_thresh=0.0,
    )
    result = session.run_tool_call(json.dumps({
        "name": "query_point_depth",
        "args": {"objects": [{"name": "target", "point": [500, 500]}]},
    }))
    payload = json.loads(result.removeprefix("<|tool_result_start|>").removesuffix("<|tool_result_end|>"))
    assert payload["target"]["depth_m"] == 3.0
