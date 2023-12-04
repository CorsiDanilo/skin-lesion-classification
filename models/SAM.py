from torch import nn
import torch
from transformers import SamProcessor, SamModel, SamConfig, SamImageProcessor


class SAM(nn.Module):
    def __init__(self, img_size=128, custom_size=False):
        super(SAM, self).__init__()
        self.img_size = img_size
        config: SamConfig = SamConfig.from_pretrained(
            "facebook/sam-vit-base")
        config.vision_config.image_size = img_size if custom_size else 1024
        config.prompt_encoder_config.image_size = img_size if custom_size else 1024
        config.prompt_encoder_config.image_embedding_size = config.prompt_encoder_config.image_size // config.vision_config.patch_size
        self.model: SamModel = SamModel.from_pretrained(
            "facebook/sam-vit-base", config=config, ignore_mismatched_sizes=True)
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
            # print(f"{name} requires grad: {param.requires_grad}")
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                if any([block in name for block in incompatible_blocks]):
                    continue
                param.requires_grad_(False)
            # TODO: with custom image size the vision encoder checkpoint are not loaded because of mismatched sizes.
            # if name.startswith("prompt_encoder"):
            #     param.requires_grad_(False)

    def forward(self, images, bboxes):
        inputs = self.processor(
            images, input_boxes=[bboxes.tolist()], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # print(f"Inputs are {inputs}")

        images = inputs["pixel_values"]
        bboxes = inputs["input_boxes"].unsqueeze(1)
        original_sizes = inputs["original_sizes"]
        reshaped_input_sizes = inputs["reshaped_input_sizes"]

        low_res_masks = self.model(pixel_values=images,
                                   input_boxes=bboxes,
                                   multimask_output=False)
        upscaled_masks = self.processor.post_process_masks(
            low_res_masks.pred_masks, original_sizes, reshaped_input_sizes, binarize=False)

        return torch.stack(upscaled_masks, dim=0).squeeze(1)

    def get_img_size(self):
        return self.img_size

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(
    #                 m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
