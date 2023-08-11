import sys

from src.data.annotation import Annotation
from src.data.bbox import Bbox
from src.data.dataset_builder import DatasetBuilder
from src.data.image_dataset import ImageDataset
from src.data.dataset_builder_config import DatasetBuilderConfig


def test_init():
    sys.argv = [""]
    config = DatasetBuilderConfig.parse_args()
    dataset_builder = DatasetBuilder(
        train_data_dir="tests/fixtures/train",
        val_data_dir="tests/fixtures/validation",
        config=config,
    )
    assert dataset_builder.train_data_dir == "tests/fixtures/train"
    assert dataset_builder.val_data_dir == "tests/fixtures/validation"
    assert dataset_builder.train_annotations == []
    assert dataset_builder.val_annotations == []


def test_get_data():
    sys.argv = [""]
    config = DatasetBuilderConfig.parse_args()
    dataset_builder = DatasetBuilder(
        train_data_dir="tests/fixtures/train",
        val_data_dir="tests/fixtures/validation",
        config=config,
    )
    annotations_list = dataset_builder._get_data(
        data_dir="tests/fixtures/train",
        annotations_split=dataset_builder.train_annotations,
    )
    assert isinstance(annotations_list, ImageDataset)
    assert isinstance(annotations_list.annotations[1], Annotation)
    assert isinstance(annotations_list.annotations[2].bboxes[0], Bbox)
    assert len(annotations_list) == 4


def test_parse_data_when_train():
    sys.argv = [""]
    config = DatasetBuilderConfig.parse_args()
    dataset_builder = DatasetBuilder(
        train_data_dir="tests/fixtures/train",
        val_data_dir="tests/fixtures/validation",
        config=config,
    )
    annotations = dataset_builder._parse_data(
        data_dir="tests/fixtures/train",
        annotations_split=dataset_builder.train_annotations,
    )
    assert isinstance(annotations, list)
    assert isinstance(annotations[3], Annotation)
    assert isinstance(annotations[3].bboxes[0], Bbox)
    assert len(annotations) == 4
    assert annotations[3].image_path == "tests/fixtures/train/images/scratches_1.jpg"
    assert annotations[3].bboxes[0].top_left_x == 26.0
    assert annotations[3].bboxes[0].top_left_y == 12.0
    assert annotations[3].bboxes[0].bottom_right_x == 43.0
    assert annotations[3].bboxes[0].bottom_right_y == 171.0
    assert annotations[3].bboxes[1].top_left_x == 8.0
    assert annotations[3].bboxes[1].top_left_y == 184.0
    assert annotations[3].bboxes[1].bottom_right_x == 17.0
    assert annotations[3].bboxes[1].bottom_right_y == 196.0


def test_parse_data_when_val():
    sys.argv = [""]
    config = DatasetBuilderConfig.parse_args()
    dataset_builder = DatasetBuilder(
        train_data_dir="tests/fixtures/train",
        val_data_dir="tests/fixtures/validation",
        config=config,
    )
    annotations = dataset_builder._parse_data(
        data_dir="tests/fixtures/validation",
        annotations_split=dataset_builder.val_annotations,
    )
    assert isinstance(annotations, list)
    assert isinstance(annotations[0], Annotation)
    assert len(annotations) == 4
    assert (
        annotations[3].image_path
        == "tests/fixtures/validation/images/scratches_242.jpg"
    )
    assert annotations[3].bboxes[0].top_left_x == 7.0
    assert annotations[3].bboxes[0].top_left_y == 35.0
    assert annotations[3].bboxes[0].bottom_right_x == 108.0
    assert annotations[3].bboxes[0].bottom_right_y == 52.0
    assert annotations[3].bboxes[1].top_left_x == 82.0
    assert annotations[3].bboxes[1].top_left_y == 156.0
    assert annotations[3].bboxes[1].bottom_right_x == 199.0
    assert annotations[3].bboxes[1].bottom_right_y == 171.0
    assert annotations[3].bboxes[2].top_left_x == 2.0
    assert annotations[3].bboxes[2].top_left_y == 169.0
    assert annotations[3].bboxes[2].bottom_right_x == 27.0
    assert annotations[3].bboxes[2].bottom_right_y == 195.0
