from torch import nn
from .sam_builder.build_sam import sam_model_registry


class SAM(nn.Module):
    def __init__(self, img_size=64):
        super(SAM, self).__init__()
        # NOTE: checkpoint available at https://huggingface.co/ybelkada/segment-anything/tree/main/checkpoints
        self.img_size = img_size
        self.model = sam_model_registry['vit_b'](
            checkpoint='checkpoints/sam_vit_b_01ec64.pth', custom_img_size=img_size)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = BoxOnlyPredictor(in_features)

    def forward(self, inputs):
        return self.model(inputs)

    def get_img_size(self):
        return self.img_size

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
