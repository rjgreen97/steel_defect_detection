import torch

from src.data.bbox import Bbox


def test_init_():
    bbox = Bbox(top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=4)
    assert bbox.top_left_x == 1
    assert bbox.top_left_y == 2
    assert bbox.bottom_right_x == 3
    assert bbox.bottom_right_y == 4
    assert bbox.label == 1


def test_coordinates():
    bbox = Bbox(top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=4)
    assert bbox.coordinates() == [1.0, 2.0, 3.0, 4.0]


def test_to_tensor():
    bbox = Bbox(top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=4)
    bbox_tensor = bbox.to_tensor()
    assert type(bbox_tensor) == torch.Tensor
    assert bbox.to_tensor().tolist() == [1.0, 2.0, 3.0, 4.0]
