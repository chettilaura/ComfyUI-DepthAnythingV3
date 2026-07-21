# Fork notes

This is a fork of [PozzettiAndrea/ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3) (MIT licensed), based on a snapshot of the project taken before its more recent restructuring upstream (the current upstream `main` has since moved to a different module layout with streaming/Triton kernels not present here).

## What changed, and why

The model loader (`nodes/nodes_loader.py`) was rewritten to:

- **Resolve model files through ComfyUI's `folder_paths` registry** instead of a fixed path, so it plays well with custom model directories and `extra_model_paths.yaml`.
- **Dynamically construct the correct network architecture** (monocular / metric / nested Gaussian variants) from a config matched against the checkpoint filename, instead of assuming a single fixed architecture.
- **Use `accelerate`'s meta-device loading** (`init_empty_weights` + `set_module_tensor_to_device`) to materialize weights directly on the target device/dtype during loading, avoiding a redundant CPU copy of the full model — useful for the larger ViT-G variants.

See [`dev-history/`](dev-history/) for the two earlier iterations of this loader, kept for reference.

## Credits

- Original implementation: [PozzettiAndrea/ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3) (MIT license)
- Underlying model: [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (ByteDance Seed Team)
