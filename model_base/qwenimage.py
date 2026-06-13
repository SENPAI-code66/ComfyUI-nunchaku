"""
Nunchaku Qwen-Image model base.

This module provides a wrapper for ComfyUI's Qwen-Image model base.
"""

import torch
from comfy.model_base import ModelType, QwenImage

from nunchaku.models.linear import SVDQW4A4Linear

from ..models.qwenimage import NunchakuQwenImageTransformer2DModel


class NunchakuQwenImage(QwenImage):
    """
    Wrapper for the Nunchaku Qwen-Image model.

    Parameters
    ----------
    model_config : object
        Model configuration object.
    model_type : ModelType, optional
        Type of the model (default is ModelType.FLUX).
    device : torch.device or str, optional
        Device to load the model onto.
    """

    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        """
        Initialize the NunchakuQwenImage model.

        Parameters
        ----------
        model_config : object
            Model configuration object.
        model_type : ModelType, optional
            Type of the model (default is ModelType.FLUX).
        device : torch.device or str, optional
            Device to load the model onto.
        """
        super(QwenImage, self).__init__(
            model_config, model_type, device=device, unet_model=NunchakuQwenImageTransformer2DModel
        )
        self.memory_usage_factor_conds = ("ref_latents",)

    def load_model_weights(self, sd: dict[str, torch.Tensor], unet_prefix: str = ""):
        """
        Load model weights into the diffusion model.

        Parameters
        ----------
        sd : dict of str to torch.Tensor
            State dictionary containing model weights.
        unet_prefix : str, optional
            Prefix for UNet weights (default is "").

        Raises
        ------
        ValueError
            If a required key is missing from the state dictionary.
        """
        diffusion_model = self.diffusion_model
        state_dict = diffusion_model.state_dict()
        for k in state_dict.keys():
            if k not in sd:
                if ".wcscales" not in k:
                    raise ValueError(f"Key {k} not found in state_dict")
                sd[k] = torch.ones_like(state_dict[k])
        for n, m in diffusion_model.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                if m.wtscale is not None:
                    m.wtscale = sd.pop(f"{n}.wtscale", 1.0)
        print("--- DEBUG DTYPES ---")
        for name, param in diffusion_model.named_parameters():
            if "wscales" in name or "wzeros" in name or "proj_" in name or "img_mod" in name or "txt_mod" in name:
                print(f"Model param {name} dtype: {param.dtype}")
        for k in list(sd.keys()):
            if "wscales" in k or "wzeros" in k or "proj_" in k or "img_mod" in k or "txt_mod" in k:
                print(f"Checkpoint weight {k} dtype: {sd[k].dtype}")
        has_fp16 = any(p.dtype == torch.float16 for p in diffusion_model.parameters())
        print(f"has_fp16: {has_fp16}")
        if has_fp16:
            from nunchaku.models.transformers.utils import convert_fp16
            convert_fp16(diffusion_model, sd)
        print("--- AFTER CONVERT DTYPES ---")
        for k in list(sd.keys()):
            if "wscales" in k or "wzeros" in k or "proj_" in k or "img_mod" in k or "txt_mod" in k:
                print(f"Checkpoint weight {k} dtype: {sd[k].dtype}")
        diffusion_model.load_state_dict(sd, strict=True)
        sd.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
