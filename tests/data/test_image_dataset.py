from src.data.annotation import Annotation
from src.data.bbox import Bbox
from src.data.image_dataset import ImageDataset


def test_init_when_annotations_is_empty_list():
    annotations = []
    image_dataset = ImageDataset(annotations=annotations)
    assert image_dataset.annotations == annotations
    assert isinstance(image_dataset.annotations, list)


def test_init_when_annotations_contains_annotations():
    annotation = Annotation(image_path="image_path")
    annotations = [annotation]
    image_dataset = ImageDataset(annotations=annotations)
    assert image_dataset.annotations == annotations
    assert isinstance(image_dataset.annotations, list)


def test_len():
    annotation = Annotation(image_path="image_path")
    annotations = [annotation]
    image_dataset = ImageDataset(annotations=annotations)
    assert len(image_dataset) == 1


def test_getitem():
    annotation = Annotation(image_path="image_path", bboxes=[Bbox(1, 2, 3, 4)])
    assert annotation.image_path == "image_path"
    assert len(annotation.bboxes) == 1
    assert annotation.bboxes[0].top_left_x == 1
    assert annotation.bboxes[0].top_left_y == 2
    assert annotation.bboxes[0].bottom_right_x == 3
    assert annotation.bboxes[0].bottom_right_y == 4
