from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch


class Model:
    @classmethod
    def get_model(cls):
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        return model

    @classmethod
    def get_model_from_checkpoint(cls, checkpoint_path):
        model = cls.get_model()
        model.load_state_dict(torch.load(checkpoint_path))
        return model
