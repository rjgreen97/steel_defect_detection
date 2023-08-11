import os


class DataPreprocessor:
    def __init__(self, train_data_dir, val_data_dir):
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir

    def preprocess_data(self) -> None:
        self._organize_data(data_dir=self.train_data_dir)
        self._organize_data(data_dir=self.val_data_dir)
        self._remove_ds_stores(data_dir=self.train_data_dir)
        self._remove_ds_stores(data_dir=self.val_data_dir)
        self._clean_data()

    def _organize_data(self, data_dir):
        for directory in os.listdir(os.path.join(data_dir, "images")):
            sub_dirs = os.path.join(data_dir, "images", directory)
            if not os.path.isdir(sub_dirs):
                continue
            for file in os.listdir(sub_dirs):
                os.rename(
                    os.path.join(sub_dirs, file),
                    os.path.join(data_dir, "images", file),
                )
            os.rmdir(sub_dirs)

    def _clean_data(self) -> None:
        """
        'crazing_240.jpg' is erroneously placed in the train dataset upon download.
        """

        os.rename(
            "data/train/images/crazing_240.jpg",
            "data/validation/images/crazing_240.jpg",
        )

    def _remove_ds_stores(self, data_dir):
        for dir in os.listdir(data_dir):
            if dir == ".DS_Store":
                os.remove(os.path.join(data_dir, dir))


if __name__ == "__main__":
    data_preprocessor = DataPreprocessor(
        train_data_dir="data/train",
        val_data_dir="data/validation",
    )
    data_preprocessor.preprocess_data()
