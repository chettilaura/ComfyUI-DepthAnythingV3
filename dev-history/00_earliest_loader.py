"""Model loading and configuration nodes for DepthAnythingV3 (Local with Debug Logs)."""

import os
from contextlib import nullcontext

import comfy.model_management as mm
import folder_paths
import torch
from comfy.utils import load_torch_file

# Relative imports to reach the sibling depth_anything_v3 folder
from .depth_anything_v3.configs import MODEL_CONFIGS
from .depth_anything_v3.model.cam_dec import CameraDec
from .depth_anything_v3.model.cam_enc import CameraEnc
from .depth_anything_v3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net
from .depth_anything_v3.model.dinov2.dinov2 import DinoV2
from .depth_anything_v3.model.dpt import DPT
from .depth_anything_v3.model.dualdpt import DualDPT
from .depth_anything_v3.model.gs_adapter import GaussianAdapter
from .depth_anything_v3.model.gsdpt import GSDPT
from .utils import DEFAULT_PATCH_SIZE, logger



#EDITED

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except (ImportError, ModuleNotFoundError):
    is_accelerate_available = False

class DownloadAndLoadDepthAnythingV3Model:
    @classmethod
    def INPUT_TYPES(s):
        print(f"\n[DA3 DEBUG] --- Rez Environment Initialization ---")
        
        # Get the colon-separated list of paths from Rez
        rez_env_val = os.environ.get("COMFY_MODEL_TYPE_DEPTHANYTHING3_MODELS", "")
        
        search_dirs = []
        if rez_env_val:
            search_dirs = rez_env_val.split(os.pathsep)
            print(f"[DA3 DEBUG] Rez variable found with {len(search_dirs)} model paths.")
        
        # Also include any paths registered via extra_model_paths.yaml
        extra_paths = folder_paths.get_folder_paths("depthanything3_models")
        if extra_paths:
            search_dirs.extend(extra_paths)
        
        local_files = []
        # Scan each specific model folder provided by the pipeline
        for root_dir in set(search_dirs):
            if not root_dir: continue
            abs_root = os.path.abspath(root_dir)
            
            if os.path.exists(abs_root):
                # Look for .safetensors inside the versioned folder
                files = [f for f in os.listdir(abs_root) if f.endswith(".safetensors")]
                if files:
                    print(f"[DA3 DEBUG] Scanned: {abs_root} | Found: {files}")
                    for f in files:
                        # We use a combined name to avoid collisions if two folders have 'model.safetensors'
                        # This shows as "folder_name/file_name" in the ComfyUI menu
                        display_name = os.path.join(os.path.basename(abs_root), f)
                        local_files.append(display_name)
            else:
                print(f"[DA3 DEBUG] Path from Rez does not exist: {abs_root}")

        local_files = sorted(list(set(local_files)))
        
        if not local_files:
            local_files = ["No models found - Check Rez folders"]

        return {
            "required": {
                "model": (local_files, {"default": local_files[0] if local_files else ""}),
            },
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("DA3MODEL",)
    RETURN_NAMES = ("da3_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "DepthAnythingV3"

    def loadmodel(self, model, precision="auto"):
        device = mm.get_torch_device()
        
        # Resolve Path from Rez/Extra Paths
        rez_env_val = os.environ.get("COMFY_MODEL_TYPE_DEPTHANYTHING3_MODELS", "")
        raw_dirs = rez_env_val.split(os.pathsep) if rez_env_val else []
        raw_dirs.extend(folder_paths.get_folder_paths("depthanything3_models"))
        
        model_path = None
        for root_dir in set([os.path.abspath(p) for p in raw_dirs if p.strip()]):
            # Check for folder/file structure
            potential_path = os.path.join(os.path.dirname(root_dir), model)
            if os.path.exists(potential_path):
                model_path = potential_path
                break
            # Check for direct file in root_dir
            potential_path = os.path.join(root_dir, os.path.basename(model))
            if os.path.exists(potential_path):
                model_path = potential_path
                break
        
        if not model_path:
            raise FileNotFoundError(f"Could not locate model file: {model}")

        print(f"[DA3 DEBUG] SUCCESSFULLY LOADING: {model_path}")

        # Define the 'config' variable
        # takes the filename (e.g., da3_giant_1.1.safetensors) and turn it into a key (da3-giant)
        filename = os.path.basename(model).lower()
        model_key = filename.replace(".safetensors", "").replace("_", "-")
        
        config = None
        # Try to find a match in the MODEL_CONFIGS dictionary
        for k in MODEL_CONFIGS:
            if k in model_key:
                config = MODEL_CONFIGS[k]
                break
        
        if config is None:
            raise ValueError(f"No configuration found for model key: {model_key} (from file {filename})")

        # Determine precision
        if precision == "auto":
            dtype = torch.float16 if "fp16" in filename else torch.float32
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Accelerate availability check
        try:
            from accelerate import init_empty_weights
            is_accel = True
        except:
            is_accel = False

#EDITED


        # Build the model architecture
        # Only use init_empty_weights on CUDA devices to avoid meta device issues on CPU
        use_empty_weights = is_accelerate_available and device.type == "cuda"

        # Encoder embed dimensions for camera modules
        encoder_embed_dims = {
            "vits": 384,
            "vitb": 768,
            "vitl": 1024,
            "vitg": 1536,
        }

        with init_empty_weights() if use_empty_weights else nullcontext():
            # Check if this is a nested model (requires two branches)
            is_nested = config.get("is_nested", False)

            if is_nested:
                logger.info(
                    "Creating nested model with main (Giant) and metric (Large) branches"
                )

                # Main branch: DA3-Giant with camera support
                backbone_main = DinoV2(
                    name=config["encoder"],  # vitg
                    out_layers=config.get("out_layers", [19, 27, 33, 39]),
                    alt_start=config.get("alt_start", 13),
                    qknorm_start=config.get("qknorm_start", 13),
                    rope_start=config.get("rope_start", 13),
                    cat_token=config.get("cat_token", True),
                )
                head_main = DualDPT(
                    dim_in=config["dim_in"],  # 3072 (vitg with cat_token)
                    output_dim=2,
                    features=config["features"],
                    out_channels=config["out_channels"],
                )
                embed_dim = encoder_embed_dims.get(config["encoder"], 1536)
                cam_enc_main = CameraEnc(
                    dim_out=embed_dim,
                    dim_in=9,
                    trunk_depth=4,
                    num_heads=embed_dim // 64,
                    mlp_ratio=4,
                    init_values=0.01,
                )
                cam_dec_main = CameraDec(dim_in=config["dim_in"])

                # Build GS modules for Giant (nested model uses Giant as main branch)
                gs_head_main, gs_adapter_main = _build_gs_modules(config)

                da3_main = DepthAnything3Net(
                    net=backbone_main,
                    head=head_main,
                    cam_dec=cam_dec_main,
                    cam_enc=cam_enc_main,
                    gs_head=gs_head_main,
                    gs_adapter=gs_adapter_main,
                )

                # Metric branch: DA3Metric-Large (no camera support, DPT head)
                metric_config = MODEL_CONFIGS.get(
                    "da3metric-large",
                    {
                        "encoder": "vitl",
                        "features": 256,
                        "out_channels": [256, 512, 1024, 1024],
                        "dim_in": 1024,
                        "out_layers": [4, 11, 17, 23],
                    },
                )
                backbone_metric = DinoV2(
                    name=metric_config.get("encoder", "vitl"),
                    out_layers=metric_config.get("out_layers", [4, 11, 17, 23]),
                    alt_start=-1,
                    qknorm_start=-1,
                    rope_start=-1,
                    cat_token=False,
                )
                head_metric = DPT(
                    dim_in=metric_config.get("dim_in", 1024),
                    output_dim=1,
                    features=metric_config.get("features", 256),
                    out_channels=metric_config.get(
                        "out_channels", [256, 512, 1024, 1024]
                    ),
                )
                da3_metric = DepthAnything3Net(
                    net=backbone_metric,
                    head=head_metric,
                    cam_dec=None,
                    cam_enc=None,
                    gs_head=None,
                    gs_adapter=None,
                )

                inner_model = NestedModelWrapper(da3_main, da3_metric)
            else:
                # Standard single-branch model
                # Create backbone (DinoV2)
                backbone = DinoV2(
                    name=config["encoder"],
                    out_layers=config.get("out_layers", [4, 11, 17, 23]),
                    alt_start=config.get("alt_start", -1),
                    qknorm_start=config.get("qknorm_start", -1),
                    rope_start=config.get("rope_start", -1),
                    cat_token=config.get("cat_token", False),
                )

                # Create head
                if config.get("is_mono", False) or config.get("is_metric", False):
                    # Use DPT head for mono/metric models
                    head = DPT(
                        dim_in=config["dim_in"],
                        output_dim=1,
                        features=config["features"],
                        out_channels=config["out_channels"],
                    )
                else:
                    # Use DualDPT for main series models
                    head = DualDPT(
                        dim_in=config["dim_in"],
                        output_dim=2,
                        features=config["features"],
                        out_channels=config["out_channels"],
                    )

                # Create camera encoder/decoder if model has camera support
                cam_enc = None
                cam_dec = None
                if config.get("has_cam", False) and config.get("alt_start", -1) != -1:
                    embed_dim = encoder_embed_dims.get(config["encoder"], 1024)
                    # Camera encoder: encodes known camera params to tokens
                    cam_enc = CameraEnc(
                        dim_out=embed_dim,
                        dim_in=9,  # 9D pose encoding: [T(3), quat(4), fov(2)]
                        trunk_depth=4,
                        num_heads=embed_dim // 64,  # Match head dim = 64
                        mlp_ratio=4,
                        init_values=0.01,
                    )
                    # Camera decoder: decodes features to camera pose
                    cam_dec = CameraDec(
                        dim_in=config["dim_in"],  # Uses concatenated token dimension
                    )

                # Build GS modules only for Giant model (it's the only one with gs_head in checkpoint)
                gs_head = None
                gs_adapter = None
                if model_key == "da3-giant":
                    gs_head, gs_adapter = _build_gs_modules(config)
                    logger.info(
                        "Built GS head and adapter for Giant model (Gaussian splatting enabled)"
                    )

                # Create the full model with camera encoder/decoder
                inner_model = DepthAnything3Net(
                    net=backbone,
                    head=head,
                    cam_dec=cam_dec,
                    cam_enc=cam_enc,
                    gs_head=gs_head,
                    gs_adapter=gs_adapter,
                )

        # Load weights
        state_dict = load_torch_file(model_path)

        # Strip 'model.' prefix from keys if present
        new_state_dict = {}
        stripped_count = 0
        for key, value in state_dict.items():
            new_key = key
            # Strip model. prefix only
            if new_key.startswith("model."):
                new_key = new_key[6:]  # Remove 'model.' prefix
                stripped_count += 1
            new_state_dict[new_key] = value

        if stripped_count > 0:
            logger.debug(f"Stripped 'model.' prefix from {stripped_count} keys")
        # Show example keys
        sample_keys = list(new_state_dict.keys())[:3]
        logger.debug(f"Sample checkpoint keys: {sample_keys}")
        # Show head keys to understand structure
        head_keys = [k for k in new_state_dict.keys() if "head." in k]
        logger.debug(f"Checkpoint head keys ({len(head_keys)} total): {head_keys[:10]}")

        # Check if checkpoint uses da3. prefix (nested model format)
        has_da3_prefix = any(k.startswith("da3.") for k in new_state_dict.keys())

        if is_nested:
            # Nested model already has da3. and da3_metric. structure
            logger.debug("Using nested model wrapper (da3 + da3_metric branches)")
            self.model = inner_model
        elif has_da3_prefix:
            # Wrap model to match nested checkpoint structure (da3.backbone... etc)
            logger.debug("Detected nested model checkpoint format (da3. prefix)")
            self.model = DA3ModelWrapper(inner_model)
        else:
            # Use model directly (keys match backbone.*, head.*)
            logger.debug("Detected standard model checkpoint format (no prefix)")
            self.model = inner_model

        if use_empty_weights:
            # Used init_empty_weights, must use set_module_tensor_to_device
            failed_keys = []
            loaded_keys = []
            for key in new_state_dict:
                try:
                    set_module_tensor_to_device(
                        self.model,
                        key,
                        device=device,
                        dtype=dtype,
                        value=new_state_dict[key],
                    )
                    loaded_keys.append(key)
                except Exception as e:
                    failed_keys.append((key, str(e)))
            if failed_keys:
                logger.warning(
                    f"Could not load {len(failed_keys)} weights (this is normal for simplified models)"
                )
                # Debug: show first few failed keys to understand the pattern
                logger.debug(
                    f"First 10 failed keys: {[k for k, e in failed_keys[:10]]}"
                )
                # Show head-specific failures
                head_failures = [
                    (k, e) for k, e in failed_keys if k.startswith("head.")
                ]
                if head_failures:
                    logger.debug(
                        f"Head failures ({len(head_failures)}): {head_failures[:5]}"
                    )

            # Materialize any remaining meta tensors (parameters not in checkpoint)
            meta_params = []
            for name, param in self.model.named_parameters():
                if param.device.type == "meta":
                    meta_params.append(name)
                    # Check if this key exists in checkpoint but wasn't loaded (shape mismatch?)
                    if name in new_state_dict:
                        ckpt_shape = new_state_dict[name].shape
                        model_shape = param.shape
                        if ckpt_shape != model_shape:
                            logger.debug(
                                f"Shape mismatch for {name}: checkpoint {ckpt_shape} vs model {model_shape}"
                            )
                    # Initialize with zeros and move to correct device
                    set_module_tensor_to_device(
                        self.model,
                        name,
                        device=device,
                        dtype=dtype,
                        value=torch.zeros(param.shape, dtype=dtype),
                    )

            if meta_params:
                logger.warning(
                    f"Initialized {len(meta_params)} missing parameters with zeros (not in checkpoint)"
                )
                # Debug: show first few meta params to understand the pattern
                logger.debug(f"First 10 missing params: {meta_params[:10]}")
        else:
            # Standard model loading (CPU or no accelerate)
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                logger.warning(f"Exception during model loading: {e}")
                # Try partial loading
                model_dict = self.model.state_dict()
                filtered_dict = {
                    k: v
                    for k, v in new_state_dict.items()
                    if k in model_dict and model_dict[k].shape == v.shape
                }
                model_dict.update(filtered_dict)
                self.model.load_state_dict(model_dict)

        # Move to device if we didn't use init_empty_weights
        if not use_empty_weights:
            self.model.to(device).to(dtype)

        self.model.eval()

        da3_model = {
            "model": self.model,
            "dtype": dtype,
            "config": config,
        }

        return (da3_model,)


class DA3_EnableTiledProcessing:
    """Configure model for tiled processing to handle high-resolution images."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL",),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 256, "max": 2048, "step": 14},
                ),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 14}),
            },
        }

    RETURN_TYPES = ("DA3MODEL",)
    RETURN_NAMES = ("da3_model",)
    FUNCTION = "configure"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Enable tiled processing for memory-efficient inference on high-resolution images.

This node configures the model to process images in tiles with overlapping regions,
then blends the results for seamless output.

Parameters:
- tile_size: Size of each tile (should be multiple of 14 for patch alignment)
- overlap: Overlap between adjacent tiles for smooth blending

Use this when:
- Processing 4K+ resolution images
- GPU memory is limited
- Getting out-of-memory errors

Note: Tiled processing may produce slightly different results at tile boundaries,
but the overlap and blending minimize artifacts.
"""

    def configure(self, da3_model, tile_size=512, overlap=64):
        # Ensure tile_size is multiple of patch size
        patch_size = DEFAULT_PATCH_SIZE
        tile_size = (tile_size // patch_size) * patch_size
        if tile_size < patch_size:
            tile_size = patch_size

        # Ensure overlap is multiple of patch size
        overlap = (overlap // patch_size) * patch_size

        # Create a copy of the model dict with tiled config
        tiled_model = {
            "model": da3_model["model"],
            "dtype": da3_model["dtype"],
            "config": da3_model["config"],
            "tiled_config": {
                "enabled": True,
                "tile_size": tile_size,
                "overlap": overlap,
            },
        }

        logger.info(
            f"Enabled tiled processing: tile_size={tile_size}, overlap={overlap}"
        )

        return (tiled_model,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadDepthAnythingV3Model": DownloadAndLoadDepthAnythingV3Model,
    "DA3_EnableTiledProcessing": DA3_EnableTiledProcessing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadDepthAnythingV3Model": "(down)Load Depth Anything V3 Model",
    "DA3_EnableTiledProcessing": "DA3 Enable Tiled Processing",
}
