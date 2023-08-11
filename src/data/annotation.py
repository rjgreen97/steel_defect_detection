import torch
import torchvision.transforms as transforms
from PIL import Image

from src.data.bbox import Bbox


class Annotation:
    def __init__(self, image_path: str = None, bboxes: list[Bbox] = None):
        self.image_path = image_path
        self.bboxes = bboxes or []

    def image(self) -> torch.Tensor:
        image = Image.open(self.image_path)
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image)

    def targets(self) -> dict:
        boxes = []
        labels = []
        for bbox in self.bboxes:
            boxes.append(bbox.to_tensor())
            labels.append(torch.tensor([bbox.label]))
        if len(boxes) > 0:
            boxes_tensor = torch.stack(boxes, axis=0)
        else:
            boxes_tensor = torch.zeros((0, 4))
        return {"boxes": boxes_tensor, "labels": torch.tensor(labels).long()}
