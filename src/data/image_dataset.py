from torch.utils.data import Dataset

from src.data.annotation import Annotation


class ImageDataset(Dataset):
    def __init__(self, annotations: list[Annotation] = None):
        self.annotations = annotations or []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        return annotation.image(), annotation.targets()
