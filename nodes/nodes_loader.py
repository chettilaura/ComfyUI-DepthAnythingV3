"""Model loading and configuration nodes for DepthAnythingV3 (folder_paths integrated)."""

import os
import torch
from contextlib import nullcontext

import comfy.model_management as mm
import folder_paths
from comfy.utils import load_torch_file

# Relative imports to reach the sibling depth_anything_v3 folder
from .depth_anything_v3.configs import MODEL_CONFIGS
from .depth_anything_v3.model.cam_dec import CameraDec
from .depth_anything_v3.model.cam_enc import CameraEnc
from .depth_anything_v3.model.da3 import DepthAnything3Net
from .depth_anything_v3.model.dinov2.dinov2 import DinoV2
from .depth_anything_v3.model.dpt import DPT
from .depth_anything_v3.model.dualdpt import DualDPT
from .depth_anything_v3.model.gs_adapter import GaussianAdapter
from .depth_anything_v3.model.gsdpt import GSDPT
from .utils import DEFAULT_PATCH_SIZE, logger

# --- REGISTER FOLDER_PATHS CATEGORY ---
DA3_CATEGORY = "depthanything3"
if DA3_CATEGORY not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(DA3_CATEGORY, os.path.join(folder_paths.models_dir, DA3_CATEGORY))

# --- HELPERS INTEGRATI ---
def _build_gs_modules(config):
    gs_head = GSDPT(dim_in=config["dim_in"], output_dim=38, features=config["features"], out_channels=config["out_channels"])
    gs_adapter = GaussianAdapter(sh_degree=2, pred_color=False, pred_offset_depth=True, pred_offset_xy=True, gaussian_scale_min=1e-5, gaussian_scale_max=30.0)
    return gs_head, gs_adapter

class DA3ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.da3 = model
    def forward(self, *args, **kwargs): return self.da3(*args, **kwargs)

class NestedModelWrapper(torch.nn.Module):
    def __init__(self, da3_main, da3_metric):
        super().__init__(); self.da3 = da3_main; self.da3_metric = da3_metric
    def forward(self, x, **kwargs): return self.da3(x, **kwargs)

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except (ImportError, ModuleNotFoundError):
    is_accelerate_available = False

class DownloadAndLoadDepthAnythingV3Model:
    @classmethod
    def INPUT_TYPES(s):
        # Get the files registered under the depthanything3 category
        local_files = folder_paths.get_filename_list(DA3_CATEGORY)
        
        if not local_files:
            local_files = ["No models found - Place them in models/depthanything3"]

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
        
        # Resolve path via folder_paths
        model_path = folder_paths.get_full_path(DA3_CATEGORY, model)
        
        if not model_path:
            raise FileNotFoundError(f"Could not locate model file: {model}")

        print(f"[DA3 DEBUG] SUCCESSFULLY LOADING: {model_path}")

        # Configuration based on the filename
        filename = os.path.basename(model).lower()
        model_key = filename.replace(".safetensors", "").replace("_", "-")
        
        config = None
        for k in MODEL_CONFIGS:
            if k in model_key:
                config = MODEL_CONFIGS[k]
                break
        
        if config is None:
            raise ValueError(f"No configuration found for model key: {model_key} (from file {filename})")

        # Determine dtype
        if precision == "auto":
            dtype = torch.float16 if "fp16" in filename else torch.float32
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        use_empty_weights = is_accelerate_available and device.type == "cuda"
        encoder_embed_dims = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}

        with init_empty_weights() if use_empty_weights else nullcontext():
            is_nested = config.get("is_nested", False)

            if is_nested:
                logger.info("Creating nested model with main (Giant) and metric (Large) branches")
                backbone_main = DinoV2(name=config["encoder"], out_layers=config.get("out_layers", [19, 27, 33, 39]), alt_start=config.get("alt_start", 13), qknorm_start=config.get("qknorm_start", 13), rope_start=config.get("rope_start", 13), cat_token=config.get("cat_token", True))
                head_main = DualDPT(dim_in=config["dim_in"], output_dim=2, features=config["features"], out_channels=config["out_channels"])
                embed_dim = encoder_embed_dims.get(config["encoder"], 1536)
                cam_enc_main = CameraEnc(dim_out=embed_dim, dim_in=9, trunk_depth=4, num_heads=embed_dim // 64, mlp_ratio=4, init_values=0.01)
                cam_dec_main = CameraDec(dim_in=config["dim_in"])
                gs_h, gs_a = _build_gs_modules(config)
                da3_main = DepthAnything3Net(net=backbone_main, head=head_main, cam_dec=cam_dec_main, cam_enc=cam_enc_main, gs_head=gs_h, gs_adapter=gs_a)

                # Metric branch
                metric_config = MODEL_CONFIGS.get("da3metric-large", {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "dim_in": 1024, "out_layers": [4, 11, 17, 23]})
                backbone_metric = DinoV2(name=metric_config["encoder"], out_layers=metric_config["out_layers"], alt_start=-1, qknorm_start=-1, rope_start=-1, cat_token=False)
                head_metric = DPT(dim_in=metric_config["dim_in"], output_dim=1, features=metric_config["features"], out_channels=metric_config["out_channels"])
                da3_metric = DepthAnything3Net(net=backbone_metric, head=head_metric)
                inner_model = NestedModelWrapper(da3_main, da3_metric)
            else:
                backbone = DinoV2(name=config["encoder"], out_layers=config.get("out_layers", [4, 11, 17, 23]), alt_start=config.get("alt_start", -1), qknorm_start=config.get("qknorm_start", -1), rope_start=config.get("rope_start", -1), cat_token=config.get("cat_token", False))
                if config.get("is_mono", False) or config.get("is_metric", False):
                    head = DPT(dim_in=config["dim_in"], output_dim=1, features=config["features"], out_channels=config["out_channels"])
                else:
                    head = DualDPT(dim_in=config["dim_in"], output_dim=2, features=config["features"], out_channels=config["out_channels"])
                
                cam_enc = cam_dec = None
                if config.get("has_cam", False) and config.get("alt_start", -1) != -1:
                    embed_dim = encoder_embed_dims.get(config["encoder"], 1024)
                    cam_enc = CameraEnc(dim_out=embed_dim, dim_in=9, trunk_depth=4, num_heads=embed_dim // 64, mlp_ratio=4, init_values=0.01)
                    cam_dec = CameraDec(dim_in=config["dim_in"])
                
                gs_h = gs_a = None
                if model_key == "da3-giant":
                    gs_h, gs_a = _build_gs_modules(config)
                
                inner_model = DepthAnything3Net(net=backbone, head=head, cam_dec=cam_dec, cam_enc=cam_enc, gs_head=gs_h, gs_adapter=gs_a)

        # Load weights
        state_dict = load_torch_file(model_path)
        new_state_dict = { (k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items() }

        has_da3_prefix = any(k.startswith("da3.") for k in new_state_dict.keys())
        if is_nested:
            self.model = inner_model
        elif has_da3_prefix:
            self.model = DA3ModelWrapper(inner_model)
        else:
            self.model = inner_model

        if use_empty_weights:
            for key in new_state_dict:
                try:
                    set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=new_state_dict[key])
                except: pass
            
            for name, param in self.model.named_parameters():
                if param.device.type == "meta":
                    set_module_tensor_to_device(self.model, name, device=device, dtype=dtype, value=torch.zeros(param.shape, dtype=dtype))
        else:
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
            except:
                m_dict = self.model.state_dict()
                filtered = {k: v for k, v in new_state_dict.items() if k in m_dict and m_dict[k].shape == v.shape}
                m_dict.update(filtered)
                self.model.load_state_dict(m_dict)

        if not use_empty_weights:
            self.model.to(device).to(dtype)

        self.model.eval()
        return ({"model": self.model, "dtype": dtype, "config": config},)

class DA3_EnableTiledProcessing:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"da3_model": ("DA3MODEL",), "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 14}), "overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 14})}}
    RETURN_TYPES = ("DA3MODEL",)
    RETURN_NAMES = ("da3_model",)
    FUNCTION = "configure"
    CATEGORY = "DepthAnythingV3"

    def configure(self, da3_model, tile_size=512, overlap=64):
        patch_size = DEFAULT_PATCH_SIZE
        tile_size = max(patch_size, (tile_size // patch_size) * patch_size)
        overlap = (overlap // patch_size) * patch_size
        tiled_model = da3_model.copy()
        tiled_model["tiled_config"] = {"enabled": True, "tile_size": tile_size, "overlap": overlap}
        return (tiled_model,)

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadDepthAnythingV3Model": DownloadAndLoadDepthAnythingV3Model,
    "DA3_EnableTiledProcessing": DA3_EnableTiledProcessing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadDepthAnythingV3Model": "Load Depth Anything V3 Model",
    "DA3_EnableTiledProcessing": "DA3 Enable Tiled Processing",
}