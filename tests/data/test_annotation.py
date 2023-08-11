import torch

from src.data.annotation import Annotation
from src.data.bbox import Bbox


def test_init_when_bbox_contains_coordinates():
    bbox = Bbox(top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=4)
    annotation = Annotation(image_path="image_path", bboxes=bbox)
    assert annotation.image_path == "image_path"
    assert annotation.bboxes == bbox
    assert isinstance(annotation.bboxes, Bbox)


def test_init_when_bbox_is_empty_list():
    bbox = []
    annotation = Annotation(image_path="image_path", bboxes=bbox)
    assert annotation.image_path == "image_path"
    assert annotation.bboxes == bbox
    assert isinstance(annotation.bboxes, list)


def test_image():
    bbox = Bbox(top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=4)
    annotation = Annotation(
        image_path="tests/fixtures/train/images/scratches_1.jpg", bboxes=bbox
    )
    assert annotation.image_path == "tests/fixtures/train/images/scratches_1.jpg"
    assert isinstance(annotation.image(), torch.Tensor)
    assert annotation.image().shape == (3, 200, 200)


def test_target_bboxes_when_multiple_boxes():
    bboxes = [
        Bbox(top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=4),
        Bbox(top_left_x=5, top_left_y=6, bottom_right_x=7, bottom_right_y=8),
    ]
    annotation = Annotation(image_path="image_path", bboxes=bboxes)
    boxes = annotation.targets()["boxes"]
    labels = annotation.targets()["labels"]
    assert boxes.shape == (2, 4)
    assert labels.shape == (2,)
    assert boxes[0][0] == 1.0
    assert boxes[0][1] == 2.0
    assert boxes[0][2] == 3.0
    assert boxes[0][3] == 4.0
    assert labels[0] == 1
    assert boxes[1][0] == 5.0
    assert boxes[1][1] == 6.0
    assert boxes[1][2] == 7.0
    assert boxes[1][3] == 8.0
    assert labels[1] == 1


def test_target_bboxes_when_single_box():
    bboxes = [
        Bbox(top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=4),
    ]
    annotation = Annotation(image_path="image_path", bboxes=bboxes)
    boxes = annotation.targets()["boxes"]
    labels = annotation.targets()["labels"]
    assert boxes.shape == (1, 4)
    assert labels.shape == (1,)
    assert boxes[0][0] == 1.0
    assert boxes[0][1] == 2.0
    assert boxes[0][2] == 3.0
    assert boxes[0][3] == 4.0
    assert labels[0] == 1


def test_target_bboxes_when_no_boxes():
    bboxes = []
    annotation = Annotation(image_path="image_path", bboxes=bboxes)
    boxes = annotation.targets()["boxes"]
    labels = annotation.targets()["labels"]
    assert boxes.shape == (0, 4)
    assert labels.shape == (0,)
