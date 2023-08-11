from pydantic import Field, PositiveFloat, PositiveInt

from src.config.base_config import BaseConfig


class TrainingConfig(BaseConfig):
    """Training configuration settings."""

    batch_size: PositiveInt = Field(
        default=12,
        description="Number of training examples used per batch.",
    )

    epochs: PositiveInt = Field(
        default=5,
        description="Number of full passes over the training dataset.",
    )

    learning_rate: PositiveFloat = Field(
        default=0.003,
        description="Learning rate for training.",
    )

    momentum: PositiveFloat = Field(
        default=0.9,
        description="Momentum for SGD optimizer.",
    )

    weight_decay: PositiveFloat = Field(
        default=0.005,
        description="Weight decay for SGD optimizer.",
    )

    patience: PositiveInt = Field(
        default=2,
        description="Number of epochs to wait for improvement before early stopping.",
    )

    loss_threshold: PositiveFloat = Field(
        default=0.005,
        description="Threshold for loss reduction to save model checkpoint.",
    )

    model_ckpt_path: str = Field(
        default="./model_ckpts/",
        description="Path to save model checkpoints.",
    )

    train_data_dir: str = Field(
        default="data/train",
        description="Directory containing training data.",
    )

    val_data_dir: str = Field(
        default="data/validation",
        description="Directory containing validation data.",
    )
