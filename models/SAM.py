from typing import Optional
from torch import nn
import torch
from transformers import SamProcessor, SamModel, SamConfig, SamImageProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SAM(nn.Module):
    def __init__(self,
                 img_size: int = 128,
                 custom_size: bool = False,
                 checkpoint_path: Optional[str] = None):
        super(SAM, self).__init__()
        self.img_size = img_size
        self.custom_size = custom_size
        config: SamConfig = SamConfig.from_pretrained(
            "facebook/sam-vit-base")
        config.vision_config.image_size = img_size if custom_size else 1024
        config.prompt_encoder_config.image_size = img_size if custom_size else 1024
        config.prompt_encoder_config.image_embedding_size = config.prompt_encoder_config.image_size // config.vision_config.patch_size
        if checkpoint_path is not None:
            self.model: SamModel = SamModel.from_pretrained(
                checkpoint_path, config=config, ignore_mismatched_sizes=True).to(device)
        else:
            self.model: SamModel = SamModel.from_pretrained(
                "facebook/sam-vit-base", config=config, ignore_mismatched_sizes=True).to(device)

        sam_image_processor = SamImageProcessor(
            do_resize=True, do_normalize=True, size={"longest_edge": img_size if custom_size else 1024}, pad_size={
                "height": img_size if custom_size else 1024,
                "width": img_size if custom_size else 1024
            }, do_rescale=False, do_pad=not custom_size)
        self.processor: SamProcessor = SamProcessor(
            image_processor=sam_image_processor)

        incompatible_blocks = [
            "vision_encoder.pos_embed",
            "vision_encoder.layers.2.attn.rel_pos_h",
            "vision_encoder.layers.2.attn.rel_pos_w",
            "vision_encoder.layers.5.attn.rel_pos_h",
            "vision_encoder.layers.5.attn.rel_pos_w",
            "vision_encoder.layers.8.attn.rel_pos_h",
            "vision_encoder.layers.8.attn.rel_pos_w",
            "vision_encoder.layers.11.attn.rel_pos_h",
            "vision_encoder.layers.11.attn.rel_pos_w",
        ] if custom_size else []

        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                if any([block in name for block in incompatible_blocks]):
                    continue
                param.requires_grad_(False)

    def forward(self, images: torch.Tensor, bboxes: Optional[torch.Tensor] = None):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        inputs = self.processor(
            images, input_boxes=[bboxes.tolist()] if bboxes is not None else None, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        images = inputs["pixel_values"]
        if bboxes is not None:
            bboxes = inputs["input_boxes"].unsqueeze(1)
        original_sizes = inputs["original_sizes"]
        reshaped_input_sizes = inputs["reshaped_input_sizes"]

        if len(original_sizes.shape) == 1:
            original_sizes = original_sizes.unsqueeze(0)
        if len(reshaped_input_sizes.shape) == 1:
            reshaped_input_sizes = reshaped_input_sizes.unsqueeze(0)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        low_res_masks = self.model(pixel_values=images.to(device),
                                   input_boxes=bboxes.to(
                                       torch.float32).to(device) if bboxes is not None else None,
                                   multimask_output=False)
        upscaled_masks = self.processor.post_process_masks(
            low_res_masks.pred_masks, original_sizes, reshaped_input_sizes, binarize=False)  # TODO: Maybe there is an error here

        return torch.stack(upscaled_masks, dim=0).squeeze(1).to(device)

    def get_img_size(self):
        return self.img_size
