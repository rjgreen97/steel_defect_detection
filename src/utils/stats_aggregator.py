import torch
from torch.optim import SGD
from torch.utils.data import DataLoader


from src.model import Model
from src.data.dataset_builder import DatasetBuilder
from src.data.dataset_builder_config import DatasetBuilderConfig
from src.training.trainer import Trainer
from src.training.training_config import TrainingConfig


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


class TrainingSession:
    """TrainingSession is responsible for model training setup and configuration."""

    def __init__(self, training_config, dataset_builder_config):
        self.training_config = training_config
        self.dataset_builder_config = dataset_builder_config

    def run(self):
        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.configure_device()
        self.create_optimizer()
        self.create_trainer()
        self.trainer.run()

    def create_datasets(self):
        dataset_builder = DatasetBuilder(
            self.training_config.train_data_dir,
            self.training_config.val_data_dir,
            self.dataset_builder_config,
        )
        self.train_dataset, self.val_dataset = dataset_builder.build_datasets()

    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            collate_fn=collate_fn,
        )

    def create_model(self):
        self.model = Model.get_model()

    def configure_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = SGD(
            params=params,
            lr=self.training_config.learning_rate,
            momentum=self.training_config.momentum,
            weight_decay=self.training_config.weight_decay,
        )

    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            optimizer=self.optimizer,
            device=self.device,
            config=self.training_config,
        )


def main():
    training_config = TrainingConfig.parse_args()
    dataset_builder_config = DatasetBuilderConfig()
    session = TrainingSession(training_config, dataset_builder_config)
    session.run()


if __name__ == "__main__":
    main()
