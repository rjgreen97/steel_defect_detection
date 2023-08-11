import os
import xml.etree.ElementTree as ET

from src.data.annotation import Annotation
from src.data.bbox import Bbox
from src.data.image_dataset import ImageDataset
from src.data.dataset_builder_config import DatasetBuilderConfig


class DatasetBuilder:
    def __init__(self, train_data_dir, val_data_dir, config):
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.config = config
        self.train_annotations = []
        self.val_annotations = []

    def build_datasets(self) -> tuple:
        train_dataset = self._get_data(self.train_data_dir, self.train_annotations)
        val_dataset = self._get_data(self.val_data_dir, self.val_annotations)
        return train_dataset, val_dataset

    def _get_data(self, data_dir: str, annotations_split: list) -> ImageDataset:
        annotations_list = self._parse_data(data_dir, annotations_split)
        return ImageDataset(annotations_list)

    def _parse_data(self, data_dir: str, annotations_split: list) -> None:
        xml_file_list = os.listdir(os.path.join(data_dir, self.config.annotations_dir))
        for xml_file in xml_file_list:
            root, image_filepath = self._get_filepaths(data_dir, xml_file)
            bbox_list = self._append_bbox_list(root)
            annotations_split = self._get_annotion_split(
                xml_file, image_filepath, bbox_list, annotations_split
            )
        return annotations_split

    def _append_bbox_list(self, root):
        bbox_list = []
        for obj in root.findall(self.config.object_tag):
            bbox = Bbox(
                top_left_x=float(obj.find(self.config.top_left_x).text),
                top_left_y=float(obj.find(self.config.top_left_y).text),
                bottom_right_x=float(obj.find(self.config.bottom_right_x).text),
                bottom_right_y=float(obj.find(self.config.bottom_right_y).text),
            )
            bbox_list.append(bbox)
        return bbox_list

    def _get_annotion_split(
        self, xml_file, image_filepath, bbox_list, annotations_split
    ):
        if xml_file.startswith(self.config.scratches_annotation):
            annotation = Annotation(image_path=image_filepath, bboxes=bbox_list)
            annotations_split.append(annotation)
        else:
            annotation = Annotation(image_path=image_filepath)
            annotations_split.append(annotation)
        return annotations_split

    def _get_filepaths(self, data_dir: str, xml_file: str):
        xml_filepath = os.path.join(data_dir, self.config.annotations_dir, xml_file)
        root = ET.parse(xml_filepath).getroot()
        image_filename = root.find(self.config.filename_tag).text
        image_filename = self._add_missing_extensions(image_filename)
        image_filepath = os.path.join(data_dir, self.config.image_tag, image_filename)
        return root, image_filepath

    def _add_missing_extensions(self, image_filename: str) -> str:
        if not image_filename.endswith(".jpg"):
            image_filename = image_filename + ".jpg"
        return image_filename


if __name__ == "__main__":
    config = DatasetBuilderConfig.parse_args()
    dataset_builder = DatasetBuilder(
        train_data_dir="data/train",
        val_data_dir="data/validation",
        config=config,
    )
    train_dataset, val_dataset = dataset_builder.build_datasets()
