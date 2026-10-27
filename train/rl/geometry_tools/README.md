# Geometry rollout tools

These files are the parity-critical v3 tool runtime used by both trace
generation and RL rollout:

- `engine_utils/runtime_v3.py`
- `engine_utils/geometry.py`
- `engine_utils/tools.py`
- `engine_utils/templates.py`
- `engine_utils/system_prompts.py`

They are the project-owned implementation used for the final experiments.
Keeping the exact runtime avoids silent train/inference drift in coordinate
decoding, depth-window sampling, 3D unprojection, number rounding, and
tool-result serialization.

The runtime is maintained as part of GeoUG. `GEOMETRY_TOOL_ROOT` may override
it for development, but official reproduction uses this bundled copy.
It is released under Apache-2.0.

