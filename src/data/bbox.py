import torch


class Bbox:
    def __init__(
        self,
        top_left_x: float = None,
        top_left_y: float = None,
        bottom_right_x: float = None,
        bottom_right_y: float = None,
    ):
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        self.label = 1

    def coordinates(self):
        return [
            float(self.top_left_x),
            float(self.top_left_y),
            float(self.bottom_right_x),
            float(self.bottom_right_y),
        ]

    def to_tensor(self):
        return torch.Tensor(self.coordinates())
