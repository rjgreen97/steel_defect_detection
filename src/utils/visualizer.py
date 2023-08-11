import glob as glob
import random
import os

import cv2
import numpy as np
import torch

from src.model import Model


class Visualizer:
    def __init__(self, image_dir, model_ckpt):
        self.image_dir = image_dir
        self.model_ckpt = model_ckpt
        self.val_images = glob.glob(f"{self.image_dir}/*")

    def show_random_image(self):
        random_index = self._select_random_image()
        image, image_name = self._get_image_and_name(random_index)
        tensor_image, orig_image = self._convert_image_to_tensor(image)
        self._load_model()
        outputs = self._run_model_inference(tensor_image)
        draw_boxes = self._get_boxes(outputs)
        self._display_image(orig_image, draw_boxes, image_name)

    def _select_random_image(self):
        random_index = random.randint(0, len(self.val_images) - 1)
        return random_index

    def _get_image_and_name(self, random_index=None):
        image_name = self.val_images[random_index].split("/")[-1].split(".")[0]
        image = cv2.imread(self.val_images[random_index])
        return image, image_name

    def _convert_image_to_tensor(self, image=None):
        orig_image = image.copy()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(float)
        tensor_image = torch.tensor(image, dtype=torch.float)
        return tensor_image, orig_image

    def _load_model(self):
        self.model = Model.get_model_from_checkpoint(checkpoint_path=self.model_ckpt)
        self.model.eval()

    def _run_model_inference(self, tensor_image=None):
        tensor_image = torch.unsqueeze(tensor_image, 0)
        with torch.no_grad():
            outputs = self.model(tensor_image)
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        return outputs

    def _get_boxes(self, outputs, detection_threshold=0.80):
        boxes = outputs[0]["boxes"].data.numpy()
        scores = outputs[0]["scores"].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        return draw_boxes

    def _display_image(self, orig_image, draw_boxes, image_name):
        for _j, box in enumerate(draw_boxes):
            cv2.rectangle(
                orig_image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 0, 255),
                1,
            )
        cv2.imwrite(
            os.path.join(
                "visuals",
                "prediction_visualizations",
                f"{image_name}_prediction.jpg",
            ),
            orig_image,
        )


if __name__ == "__main__":
    visualizer = Visualizer(
        image_dir="data/validation/images",
        model_ckpt="model_ckpts/0607-104238_epoch_4.pth",
    )
    visualizer.show_random_image()
